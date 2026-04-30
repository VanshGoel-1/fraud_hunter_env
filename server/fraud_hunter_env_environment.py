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

import gc
import hashlib
import json
import random
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from fraud_hunter_env.data_gen.case_compiler import generate_multimodal_aks_case
from fraud_hunter_env.models import (
    ActionKind,
    FraudHunterAction,
    FraudHunterObservation,
    MAX_EPISODE_STEPS,
)
from fraud_hunter_env.schema import FILESYSTEM_EVIDENCE_DIRS, SQL_TABLES
from fraud_hunter_env.server.data_loader import CaseHandle
from fraud_hunter_env.server.difficulty import get_difficulty_manager
from fraud_hunter_env.server.grader import GraderOutput, compute_agentic_recall, grade


_DEFAULT_CASE_BANK = Path(__file__).resolve().parent.parent / "data" / "case_bank"
_TABLE_ACCESS_RE = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)


def _case_seed(case_dir: Path) -> Optional[int]:
    db_path = case_dir / "medicare_records.db"
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT value FROM case_metadata WHERE key = 'seed'"
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def _bank_cases_for_tier(
    bank_dir: Path,
    tier: int,
    seed_range: tuple[int, int] | None = None,
) -> list[Path]:
    """Return all pre-built case directories under bank_dir/tier_N that have a DB."""
    tier_dir = bank_dir / f"tier_{tier}"
    if not tier_dir.is_dir():
        return []
    candidates = [
        p for p in tier_dir.iterdir()
        if p.is_dir() and (p / "medicare_records.db").is_file()
    ]
    if seed_range is None:
        return candidates
    lo, hi = seed_range
    return [
        p for p in candidates
        if (seed := _case_seed(p)) is not None and lo <= seed <= hi
    ]


CASE_BRIEF_TEMPLATE = (
    "You are a qui tam fraud investigator operating under the False Claims Act.\n"
    "Case id: {case_id} | Difficulty: Tier {tier}\n\n"
    "A whistleblower alleges fraud involving shell companies, Medicare abuse,\n"
    "government contracting irregularities, and/or PPP loan fraud.\n\n"
    "IMPORTANT: Wrap your reasoning in <think>...</think> tags before every action.\n"
    "Actions without reasoning will be penalised.\n\n"
    "Contradiction format contract:\n"
    "  - Beneficiaries must be referenced as beneficiary:[DESYNPUF_ID]\n"
    "  - Claims must be referenced as claim:[CLM_ID]\n"
    "  - Providers can be referenced as provider_npi:[10-digit NPI]\n"
    "  Example: evidence_a='beneficiary:BENE_0001', evidence_b='claim:C_FRAUD_DEAD'\n\n"
    "Available tools:\n"
    "  query_corporate(entity_name|entity_id) → corporate registry + filings\n"
    "  query_medicare(beneficiary_id|claim_id) → claims / beneficiary records\n"
    "  sql_query(sql_statement) → raw SELECT on the case database\n"
    "  code_act(python_code)   → sandboxed Python with `conn`, `pd`, and read-only helpers for evidence files\n"
    "      Use code_act to inspect intercepted_comms/*.txt and scanned_claims/*.pdf when you need filesystem evidence.\n"
    "  extract_entity(name, kind, npi_code?) → flag an entity as fraudulent\n"
    "  link_shell(child_entity, parent_entity) → assert UBO ownership\n"
    "  claim_contradiction(evidence_a, evidence_b, contradiction_kind) → flag anomaly\n"
    "  ocr_document(pdf_path) → returns OCR text and the base64-encoded source document\n"
    "  compare_doc_vs_claim(claim_id, extracted_fields) → verify OCR fields against the claim record\n"
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

    def __init__(
        self,
        case_bank_dir: str | None = None,
        rng_seed: int | None = None,
        case_seed_range: tuple[int, int] | None = None,
        on_episode_end: Optional[Callable[[dict[str, Any]], None]] = None,
        specific_case_id: str | None = None,
        session_id: str | None = None,
    ):
        # specific_case_id: pin reset() to a single case directory by name
        #   (e.g. "t3_a1b2c3d4"). Used by GRPO so all completions in a group
        #   evaluate the same case, otherwise group-relative advantages would
        #   measure case-difficulty noise instead of policy quality.
        # session_id: stable identifier for the difficulty manager. When None
        #   we mint a fresh UUID per env (legacy behaviour); training pipelines
        #   should pass a stable id so RLVE accumulates across reward calls.
        # Manual tempdir (no auto-finalizer): tempfile.TemporaryDirectory's
        # GC-time cleanup raises noisy PermissionErrors on Windows when SQLite
        # has briefly held the medicare_records.db handle past conn.close().
        # We own the lifecycle via close()/__del__ and use a retry-then-ignore
        # rmtree so cleanup is silent and best-effort on every platform.
        self._sandbox_dir: str = tempfile.mkdtemp(prefix="fhe_")
        self._closed: bool = False
        bank = Path(case_bank_dir) if case_bank_dir else _DEFAULT_CASE_BANK
        self._bank_dir: Optional[Path] = bank if bank.is_dir() else None
        self._rng = random.Random(rng_seed)
        self._case_seed_range = case_seed_range
        self._specific_case_id = specific_case_id
        self._on_episode_end = on_episode_end
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
        self._session_id: str = session_id or str(uuid4())
        self._replay_prev_hash: str | None = None
        self._replay_hash: str | None = None

        self._diff_mgr = get_difficulty_manager()

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _update_replay_hash(
        self,
        phase: str,
        action: FraudHunterAction | None = None,
        reward: float | None = None,
        done: bool | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "phase": phase,
            "session_id": self._session_id,
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "case_id": self._case.case_id if self._case is not None else None,
            "difficulty_tier": self._difficulty_tier,
            "seed_range": self._case_seed_range,
            "queried_tables": sorted(self._queried_tables),
            "entities": sorted(self._extracted),
            "links": sorted([list(x) for x in self._linked]),
            "contradictions": sorted([list(x) for x in self._contradictions]),
            "proof_trace": list(self._proof_trace),
            "episode_reward": round(self._episode_reward, 6),
            "previous_hash": self._replay_hash,
        }
        if action is not None:
            payload["action"] = action.model_dump(exclude_none=True, mode="json")
        if reward is not None:
            payload["reward"] = round(float(reward), 6)
        if done is not None:
            payload["done"] = bool(done)

        self._replay_prev_hash = self._replay_hash
        self._replay_hash = self._hash_payload(payload)
        return self._replay_hash

    def _replay_info(self) -> dict[str, Any]:
        return {
            "replay_hash": self._replay_hash,
            "replay_prev_hash": self._replay_prev_hash,
            "replay_hash_alg": "sha256",
        }

    def _record_source_access(self, source: str) -> None:
        value = (source or "").strip().lower().replace("\\", "/")
        if not value:
            return
        for table in SQL_TABLES:
            if value == table:
                self._queried_tables.add(table)
                return
        for dirname in FILESYSTEM_EVIDENCE_DIRS:
            if value == dirname or value.startswith(dirname + "/"):
                self._queried_tables.add(dirname)
                return

    def _trace_sql_statement(self, sql: str) -> None:
        statement = (sql or "").strip().lower()
        if not statement:
            return
        for match in _TABLE_ACCESS_RE.finditer(statement):
            self._record_source_access(match.group(1))

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

        sandbox_path = Path(self._sandbox_dir)

        # Prefer pre-built bank when available; fall back to on-the-fly generation.
        bank_pick: Optional[Path] = None
        if self._bank_dir is not None:
            # Pinned case path: search every tier for the requested case_id so
            # GRPO group-mates can be re-anchored to the exact same case across
            # fresh env instances.
            if self._specific_case_id is not None:
                for t in range(1, 6):
                    candidate = self._bank_dir / f"tier_{t}" / self._specific_case_id
                    if candidate.is_dir() and (candidate / "medicare_records.db").is_file():
                        bank_pick = candidate
                        # Pinning a case implies the tier is fixed by that case.
                        self._difficulty_tier = t
                        break

            if bank_pick is None:
                candidates = _bank_cases_for_tier(
                    self._bank_dir,
                    self._difficulty_tier,
                    self._case_seed_range,
                )
                # Tier-down fallback: scan lower tiers if current tier has no cases
                t = self._difficulty_tier
                while not candidates and t > 1:
                    t -= 1
                    candidates = _bank_cases_for_tier(self._bank_dir, t, self._case_seed_range)
                if candidates:
                    bank_pick = self._rng.choice(candidates)

        if bank_pick is not None:
            case_id = bank_pick.name
            dest = sandbox_path / case_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(bank_pick, dest)
        else:
            case_id = f"case_{uuid4().hex[:8]}"
            generation_seed = None
            if self._case_seed_range is not None:
                generation_seed = self._rng.randint(*self._case_seed_range)
            generate_multimodal_aks_case(
                sandbox_path,
                case_id,
                self._difficulty_tier,
                rng_seed=generation_seed,
            )

        db_path = sandbox_path / case_id / "medicare_records.db"
        # check_same_thread=False: the CodeAct sandbox runs `exec` in a worker
        # thread (for timeout enforcement), so the connection must be usable
        # outside the thread that created it. Access is sequential (sandbox
        # joins the worker before returning), so SQLite's own thread safety
        # guarantees still hold.
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._case = CaseHandle(
            case_id=case_id, db_path=db_path, conn=conn, tier=self._difficulty_tier
        )
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
        self._replay_prev_hash = None
        self._replay_hash = None

        brief = CASE_BRIEF_TEMPLATE.format(
            case_id=self._case.case_id,
            budget=MAX_EPISODE_STEPS,
            tier=self._difficulty_tier,
        )
        self._update_replay_hash(phase="reset")
        return FraudHunterObservation(
            case_brief=brief,
            base64_document=None,
            step_count=0,
            budget_remaining=MAX_EPISODE_STEPS,
            difficulty_tier=self._difficulty_tier,
            done=False,
            reward=0.0,
            info={"case_id": self._case.case_id, **self._build_metrics(), **self._replay_info()},
        )

    def step(self, action: FraudHunterAction) -> FraudHunterObservation:  # type: ignore[override]
        if self._case is None:
            return self.reset()

        self._state.step_count += 1
        self._total_steps += 1

        # Track actual sources touched by the action itself.
        if action.kind == ActionKind.QUERY_CORPORATE:
            self._record_source_access("corporate_registry")
        elif action.kind == ActionKind.QUERY_MEDICARE:
            self._record_source_access("beneficiary_summary")
            self._record_source_access("carrier_claims")
        elif action.kind == ActionKind.SQL_QUERY:
            self._trace_sql_statement(action.sql_statement or "")
        elif action.kind in (ActionKind.OCR_DOCUMENT, ActionKind.COMPARE_DOC_VS_CLAIM):
            self._queried_tables.add("scanned_claims")
            self._queried_tables.add("evidence_documents")
            if action.kind == ActionKind.COMPARE_DOC_VS_CLAIM:
                self._record_source_access("carrier_claims")

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
            source_access_callback=self._record_source_access,
            sql_trace_callback=self._trace_sql_statement,
        )

        self._proof_trace = out.proof_trace
        self._episode_reward += out.reward

        # Track hallucinations
        if any("hallucination" in h for h in out.hits):
            self._hallucination_count += 1

        # Fold positive signals into persistent sets
        if (action.kind == ActionKind.EXTRACT_ENTITY and action.extracted_name
                and any(hit.startswith("extract=") for hit in out.hits)):
            self._extracted.add(action.extracted_name.lower())
        elif (action.kind == ActionKind.LINK_SHELL and action.child_entity and action.parent_entity
              and any(hit.startswith("shell_link=") for hit in out.hits)):
            self._linked.add((action.child_entity.lower(), action.parent_entity.lower()))
        elif (action.kind == ActionKind.CLAIM_CONTRADICTION
              and action.evidence_a and action.evidence_b
              and any(hit.startswith("contradiction=") or hit.startswith("contradiction_fuzzy=")
                      for hit in out.hits)):
            self._contradictions.add((action.evidence_a.lower(), action.evidence_b.lower()))
        elif action.kind == ActionKind.SUBMIT_CASE:
            self._submitted = True

        budget_remaining = max(0, MAX_EPISODE_STEPS - self._state.step_count)
        done = out.done or budget_remaining == 0
        self._update_replay_hash(
            phase="step",
            action=action,
            reward=out.reward,
            done=done,
        )

        # Fire metrics callback on terminal step (before next reset would clobber state)
        if done and self._on_episode_end is not None:
            try:
                self._on_episode_end({
                    "case_id": self._case.case_id,
                    "session_id": self._session_id,
                    **self._build_metrics(),
                    **self._replay_info(),
                })
            except Exception:
                # Never let a metrics emit break the agent loop.
                pass

        return FraudHunterObservation(
            tool_output=out.tool_output,
            base64_document=getattr(out, "base64_document", None),
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
                **self._replay_info(),
            },
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def case(self) -> Optional[CaseHandle]:
        """Public accessor for the active case handle (None before reset()).

        Tests and operator scripts should prefer this over reaching into
        the private ``_case`` attribute.
        """
        return self._case

    def close(self) -> None:
        """Idempotent best-effort cleanup. Safe to call multiple times.

        Closes the active SQLite connection, drops the Python reference,
        forces a GC cycle, then rmtrees the sandbox dir. On Windows the
        sqlite handle can briefly outlive ``conn.close()``, so we retry the
        rmtree a few times before falling back to ``ignore_errors=True``.
        Cleanup never raises — finalisers can't propagate exceptions and
        leaking the temp dir is preferable to a crash.
        """
        if self._closed:
            return
        self._closed = True

        # 1) Close active case → drops the SQLite conn ref.
        if self._case is not None:
            try:
                self._case.close()
            except Exception:
                pass
            self._case = None

        # 2) Force a GC cycle so any lingering sqlite3.Connection refs are
        #    released before we try to delete the underlying .db file.
        try:
            gc.collect()
        except Exception:
            pass

        # 3) Retry rmtree to win the Windows file-lock race; fall back to
        #    ignore_errors so we never raise from cleanup.
        sandbox = self._sandbox_dir
        for attempt in range(5):
            try:
                shutil.rmtree(sandbox)
                return
            except FileNotFoundError:
                return
            except (PermissionError, OSError):
                time.sleep(0.05 * (attempt + 1))
        try:
            shutil.rmtree(sandbox, ignore_errors=True)
        except Exception:
            pass

    def __del__(self):
        # Defensive: if openenv-core or a test harness drops the env without
        # calling close(), the GC path still cleans up silently.
        try:
            self.close()
        except Exception:
            pass
