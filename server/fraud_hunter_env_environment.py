"""
Fraud Hunter Env — core environment implementation (RLVE + RLVR).

One episode = one synthetic fraud case sampled from the tiered case bank.
The agent issues FraudHunterActions (with mandatory <think> CoT traces);
the RLVR grader scores each step using 7 hierarchical layers.

RLVE: The DifficultyManager tracks per-session proficiency and dynamically
escalates the difficulty tier each episode, keeping the agent at its frontier.

Evidence Graph: The environment accumulates a structured graph of confirmed
entities, shell links, and contradictions — returned in every observation
to enable the agent to reason over its own prior discoveries.
"""

from __future__ import annotations

from uuid import uuid4
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionKind, EpisodeMetrics,
        FraudHunterAction, FraudHunterObservation,
        MAX_EPISODE_STEPS, TYPOLOGY_MULTIPLIERS,
    )
except ImportError:
    from models import (  # type: ignore
        ActionKind, EpisodeMetrics,
        FraudHunterAction, FraudHunterObservation,
        MAX_EPISODE_STEPS, TYPOLOGY_MULTIPLIERS,
    )

from .data_loader import CaseBank, CaseHandle
from .grader import grade, format_gate, GraderOutput, compute_agentic_recall
from .difficulty import get_difficulty_manager
from .sandbox import execute_code, execute_sql

import tempfile
import sqlite3
from pathlib import Path
from fraud_hunter_env.data_gen.case_compiler import generate_multimodal_aks_case


CASE_BRIEF_TEMPLATE = (
    "You are a qui tam fraud investigator operating under the False Claims Act.\n"
    "Case id: {case_id} | Difficulty: Tier {tier}\n\n"
    "A whistleblower alleges fraud involving shell companies, Medicare abuse,\n"
    "government contracting irregularities, and/or PPP loan fraud.\n\n"
    "IMPORTANT: Wrap your reasoning in <think>...</think> tags before every action.\n"
    "Actions without reasoning will be penalised.\n\n"
    "Available tools:\n"
    "  query_corporate(entity_name|entity_id) → corporate registry + filings\n"
    "  query_medicare(beneficiary_id|claim_id) → claims / beneficiary records\n"
    "  sql_query(sql_statement) → raw SELECT on the case database\n"
    "  code_act(python_code)   → sandboxed Python with `conn` and `pd`\n"
    "  extract_entity(name, kind, npi_code?) → flag an entity as fraudulent\n"
    "  link_shell(child_entity, parent_entity) → assert UBO ownership\n"
    "  claim_contradiction(evidence_a, evidence_b, contradiction_kind) → flag anomaly\n"
    "  submit_case(case_summary, confidence, typologies?) → terminate and seek conviction\n\n"
    "Fraud typologies: dead_patient_claim, duplicate_bill, upcoding, unbundling,\n"
    "  aks_violation, off_label_marketing, double_billing, cost_pricing_fraud,\n"
    "  product_substitution, ppp_fraud, foreign_affiliation, phantom_beneficiary\n\n"
    "Budget: {budget} steps. Format-gate: invalid JSON → -10 + episode ends.\n"
    "NPI validation: provider extractions require exact 10-digit NPI match.\n"
)


class FraudHunterEnvironment(Environment):
    """OpenEnv Environment with RLVE difficulty adaptation and RLVR grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, case_bank_dir: str | None = None, rng_seed: int | None = None):
        self._sandbox_dir = tempfile.TemporaryDirectory()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._case: Optional[CaseHandle] = None
        self._extracted: set[str] = set()
        self._linked: set[tuple[str, str]] = set()
        self._contradictions: set[tuple[str, str]] = set()
        self._submitted: bool = False
        self._proof_trace: list[str] = []
        self._queried_tables: set[str] = set()
        self._cot_valid_steps: int = 0
        self._total_steps: int = 0
        self._format_errors: int = 0
        self._hallucination_count: int = 0
        self._episode_reward: float = 0.0
        self._difficulty_tier: int = 1
        self._session_id: str = str(uuid4())

        self._diff_mgr = get_difficulty_manager()

    def _build_evidence_graph(self) -> dict[str, Any]:
        """Construct the evidence graph returned in every observation."""
        return {
            "entities": sorted(self._extracted),
            "shell_links": [{"child": c, "parent": p} for c, p in self._linked],
            "contradictions": [{"a": a, "b": b} for a, b in self._contradictions],
            "proof_chain_length": len(self._proof_trace),
        }

    def _build_metrics(self) -> dict:
        recall = 0.0
        if self._case:
            recall = compute_agentic_recall(self._queried_tables, self._case)
        cot_validity = self._cot_valid_steps / max(self._total_steps, 1)
        return {
            "agentic_recall": round(recall, 3),
            "cot_validity_score": round(cot_validity, 3),
            "format_error_count": self._format_errors,
            "hallucination_count": self._hallucination_count,
            "proof_chain_length": len(self._proof_trace),
            "episode_reward": round(self._episode_reward, 2),
            "difficulty_tier": self._difficulty_tier,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> FraudHunterObservation:
        # Record previous episode to difficulty manager
        if self._case is not None and self._total_steps > 0:
            self._diff_mgr.record_episode(
                self._session_id, self._episode_reward, self._format_errors
            )
            self._case.close()

        # Get current tier from RLVE
        self._difficulty_tier = self._diff_mgr.get_tier(self._session_id)
        
        # On-the-fly generation directly into the sandbox workspace
        case_id = f"case_{uuid4().hex[:8]}"
        sandbox_path = Path(self._sandbox_dir.name)
        generate_multimodal_aks_case(sandbox_path, case_id, self._difficulty_tier)
        
        # Connect to the generated sandbox database
        db_path = sandbox_path / case_id / "medicare_records.db"
        conn = sqlite3.connect(str(db_path))
        self._case = CaseHandle(case_id=case_id, db_path=db_path, conn=conn, tier=self._difficulty_tier)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._extracted.clear()
        self._linked.clear()
        self._contradictions.clear()
        self._submitted = False
        self._proof_trace = []
        self._queried_tables = set()
        self._cot_valid_steps = 0
        self._total_steps = 0
        self._format_errors = 0
        self._hallucination_count = 0
        self._episode_reward = 0.0

        brief = CASE_BRIEF_TEMPLATE.format(
            case_id=self._case.case_id,
            budget=MAX_EPISODE_STEPS,
            tier=self._difficulty_tier,
        )
        return FraudHunterObservation(
            case_brief=brief,
            step_count=0,
            budget_remaining=MAX_EPISODE_STEPS,
            difficulty_tier=self._difficulty_tier,
            done=False,
            reward=0.0,
            info={"case_id": self._case.case_id, **self._build_metrics()},
        )

    def step(self, action: FraudHunterAction) -> FraudHunterObservation:  # type: ignore[override]
        if self._case is None:
            return self.reset()

        self._state.step_count += 1
        self._total_steps += 1

        # Track which tables are being queried (for agentic recall)
        if action.kind == ActionKind.QUERY_CORPORATE:
            self._queried_tables.add("corporate_registry")
        elif action.kind == ActionKind.QUERY_MEDICARE:
            self._queried_tables.update({"medicare_claims", "medicare_beneficiaries"})
        elif action.kind in (ActionKind.SQL_QUERY, ActionKind.CODE_ACT):
            # Infer queried tables from SQL or code content
            code_or_sql = (action.sql_statement or action.python_code or "").lower()
            for tbl in ["general_ledger", "referral_payments", "government_contracts",
                         "contract_invoices", "lab_results", "loan_applications",
                         "corporate_registry", "providers", "medicare_claims",
                         "medicare_beneficiaries"]:
                if tbl in code_or_sql:
                    self._queried_tables.add(tbl)

        # CoT tracking
        if action.think_trace:
            self._cot_valid_steps += 1

        out: GraderOutput = grade(
            action=action,
            case=self._case,
            extracted=self._extracted,
            linked=self._linked,
            contradictions=self._contradictions,
            submitted=self._submitted,
            step_count=self._state.step_count,
            proof_trace=self._proof_trace,
        )

        self._proof_trace = out.proof_trace
        self._episode_reward += out.reward

        # Track hallucinations
        if any("hallucination" in h for h in out.hits):
            self._hallucination_count += 1

        # Fold positive signals into persistent sets
        if action.kind == ActionKind.EXTRACT_ENTITY and action.extracted_name:
            self._extracted.add(action.extracted_name.lower())
        elif action.kind == ActionKind.LINK_SHELL and action.child_entity and action.parent_entity:
            self._linked.add((action.child_entity.lower(), action.parent_entity.lower()))
        elif (action.kind == ActionKind.CLAIM_CONTRADICTION
              and action.evidence_a and action.evidence_b):
            self._contradictions.add((action.evidence_a.lower(), action.evidence_b.lower()))
        elif action.kind == ActionKind.SUBMIT_CASE:
            self._submitted = True

        budget_remaining = max(0, MAX_EPISODE_STEPS - self._state.step_count)
        done = out.done or budget_remaining == 0

        return FraudHunterObservation(
            tool_output=out.tool_output,
            grader_feedback=out.feedback,
            evidence_graph=self._build_evidence_graph(),
            step_count=self._state.step_count,
            budget_remaining=budget_remaining,
            difficulty_tier=self._difficulty_tier,
            done=done,
            reward=out.reward,
            info={
                "hits": out.hits,
                "case_id": self._case.case_id,
                **self._build_metrics(),
            },
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def extracted(self) -> set[str]:
        return self._extracted

    @property
    def linked(self) -> set[tuple[str, str]]:
        return self._linked

    @property
    def contradictions(self) -> set[tuple[str, str]]:
        return self._contradictions
