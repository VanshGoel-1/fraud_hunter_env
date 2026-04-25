"""
Pydantic schemas for the Government Fraud Hunter AI environment.

Architecture: Format-First Hierarchical Curriculum + RLVR
- Every action MUST include <think>...</think> CoT before the JSON payload
- Format gate terminates episode on schema violation (prevents reward hacking)
- Per-kind fields are strictly validated; NPI requires exact checksum match
- Evidence graph accumulates across steps to enable causal chain scoring

Supports:
  - Standard tool actions (query_corporate, query_medicare, etc.)
  - CodeAct: agent submits Python code executed in a sandboxed environment
  - SQL_QUERY: agent issues raw SQL (MCP-style tool call)
  - Strict NPI validation (Luhn checksum, no partial credit)
  - 7 fraud typologies with per-typology reward multipliers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


# ─── Action Kinds ─────────────────────────────────────────────────────────────

class ActionKind(str, Enum):
    QUERY_CORPORATE   = "query_corporate"    # lookup corporate registry
    QUERY_MEDICARE    = "query_medicare"     # lookup beneficiary / claim
    EXTRACT_ENTITY    = "extract_entity"     # flag an entity as fraudulent
    LINK_SHELL        = "link_shell"         # assert UBO / shell relationship
    CLAIM_CONTRADICTION = "claim_contradiction"  # flag a billing anomaly
    SQL_QUERY         = "sql_query"          # MCP-style: raw restricted SQL
    CODE_ACT          = "code_act"           # CodeAct: sandboxed Python
    OCR_DOCUMENT      = "ocr_document"       # OCR a scanned PDF into field/value text
    COMPARE_DOC_VS_CLAIM = "compare_doc_vs_claim"  # verify OCR extraction against DB
    SUBMIT_CASE       = "submit_case"        # terminal: seek conviction


class EntityKind(str, Enum):
    BENEFICIARY  = "beneficiary"
    PROVIDER     = "provider"
    CORPORATION  = "corporation"
    UBO          = "ubo"          # Ultimate Beneficial Owner
    CONTRACTOR   = "contractor"   # government contracting vendor


class ContradictionKind(str, Enum):
    # Healthcare fraud
    DEAD_PATIENT_CLAIM  = "dead_patient_claim"
    DUPLICATE_BILL      = "duplicate_bill"
    UPCODING            = "upcoding"           # lower CPT billed as higher
    UNBUNDLING          = "unbundling"         # bundled services split for excess billing
    PHANTOM_BENEFICIARY = "phantom_beneficiary"
    AKS_VIOLATION       = "aks_violation"      # Anti-Kickback Statute
    OFF_LABEL_MARKETING = "off_label_marketing"
    # Government contracting fraud
    DOUBLE_BILLING      = "double_billing"     # same invoice line submitted twice
    COST_PRICING_FRAUD  = "cost_pricing_fraud" # inaccurate pricing data
    PRODUCT_SUBSTITUTION = "product_substitution" # different item delivered
    # PPP / pandemic fraud
    PPP_FRAUD           = "ppp_fraud"         # misrepresented employee count
    FOREIGN_AFFILIATION = "foreign_affiliation"  # undisclosed foreign govt ties


# ─── Episode Metrics (tracked in environment, returned in observation) ─────────

@dataclass
class EpisodeMetrics:
    """Per-episode diagnostics for training monitoring and CoT-Pass@K scoring."""
    agentic_recall: float = 0.0          # tables/entities queried / gold required
    cot_validity_score: float = 0.0      # fraction of steps with valid <think> blocks
    format_error_count: int = 0          # cumulative format gate violations
    hallucination_count: int = 0         # actions on non-existent entities
    proof_chain_complete: bool = False   # full causal chain from entity → link → contradiction
    steps_to_first_hit: int = -1         # steps until first positive reward
    typologies_found: List[str] = field(default_factory=list)


# ─── Action Schema ─────────────────────────────────────────────────────────────

class FraudHunterAction(Action):
    """
    Single polymorphic action. The LLM MUST emit:

        <think>
        I see beneficiary B001 has dod=2025-11-03. Claim C003 was filed 2026-03-01.
        This is a dead patient billing contradiction.
        </think>
        {"kind": "claim_contradiction", "evidence_a": "beneficiary:B001",
         "evidence_b": "claim:C003", "contradiction_kind": "dead_patient_claim"}

    The <think> block is parsed by the environment and scored by the CoT verifier.
    Absence of a <think> block applies COT_MISSING_PENALTY (soft, non-terminal).

    Per-kind required fields are enforced by `_require_kind_fields`.
    Schema violations (missing required fields, wrong types) trigger FORMAT_GATE_PENALTY
    and terminate the episode immediately.

    NPI validation: when extracting a provider, `npi_code` must match the ground-truth
    NPI exactly. No partial credit; mismatch → NPI_MISMATCH_PENALTY.
    """

    kind: ActionKind = Field(..., description="Action variant to dispatch")

    # Reasoning trace (CoT) — parsed from <think>...</think> wrapper
    think_trace: Optional[str] = Field(
        default=None,
        description="Agent's chain-of-thought reasoning before acting"
    )

    # query_corporate
    entity_name: Optional[str] = Field(default=None)
    entity_id:   Optional[str] = Field(default=None)

    # query_medicare
    beneficiary_id: Optional[str] = Field(default=None)
    claim_id:       Optional[str] = Field(default=None)

    # extract_entity
    extracted_name: Optional[str]       = Field(default=None)
    extracted_kind: Optional[EntityKind] = Field(default=None)
    npi_code:       Optional[str]       = Field(
        default=None, description="10-digit NPI (required for provider extraction)"
    )

    # link_shell
    child_entity:  Optional[str] = Field(default=None)
    parent_entity: Optional[str] = Field(default=None)

    # claim_contradiction
    evidence_a:          Optional[str]               = Field(default=None)
    evidence_b:          Optional[str]               = Field(default=None)
    contradiction_kind:  Optional[ContradictionKind] = Field(default=None)
    cpt_code:            Optional[str]               = Field(default=None, description="CPT code involved")
    icd10_code:          Optional[str]               = Field(default=None, description="ICD-10 diagnosis code")

    # sql_query (MCP-style)
    sql_statement: Optional[str] = Field(
        default=None, description="Restricted SQL SELECT statement"
    )

    # code_act (CodeAct paradigm)
    python_code: Optional[str] = Field(
        default=None, description="Sandboxed Python code using `conn` and `pd`"
    )

    # ocr_document — path to a scanned claim PDF in evidence_documents.pdf_path
    pdf_path: Optional[str] = Field(
        default=None, description="Filesystem path to a scanned evidence PDF"
    )

    # compare_doc_vs_claim — agent's OCR-extracted fields for verification
    extracted_fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fields the agent read off a scanned claim (e.g. {'hcpcs_code':'99215','amount':350.0})"
    )

    # submit_case
    case_summary: Optional[str]  = Field(default=None)
    confidence:   Optional[float] = Field(default=None, ge=0.0, le=1.0)
    typologies:   Optional[List[str]] = Field(default=None, description="List of fraud typologies alleged")

    @model_validator(mode="after")
    def _require_kind_fields(self) -> "FraudHunterAction":
        if self.kind == ActionKind.QUERY_CORPORATE:
            if not (self.entity_name or self.entity_id):
                raise ValueError("query_corporate requires entity_name or entity_id")

        elif self.kind == ActionKind.QUERY_MEDICARE:
            if not (self.beneficiary_id or self.claim_id):
                raise ValueError("query_medicare requires beneficiary_id or claim_id")

        elif self.kind == ActionKind.EXTRACT_ENTITY:
            if not self.extracted_name:
                raise ValueError("extract_entity requires extracted_name")
            if not self.extracted_kind:
                raise ValueError("extract_entity requires extracted_kind")
            if self.extracted_kind == EntityKind.PROVIDER and not self.npi_code:
                raise ValueError("extract_entity for provider requires npi_code (strict NPI validation)")

        elif self.kind == ActionKind.LINK_SHELL:
            if not self.child_entity:
                raise ValueError("link_shell requires child_entity")
            if not self.parent_entity:
                raise ValueError("link_shell requires parent_entity")

        elif self.kind == ActionKind.CLAIM_CONTRADICTION:
            if not self.evidence_a:
                raise ValueError("claim_contradiction requires evidence_a")
            if not self.evidence_b:
                raise ValueError("claim_contradiction requires evidence_b")
            if not self.contradiction_kind:
                raise ValueError("claim_contradiction requires contradiction_kind")

        elif self.kind == ActionKind.SQL_QUERY:
            if not self.sql_statement:
                raise ValueError("sql_query requires sql_statement")
            stmt = self.sql_statement.strip().upper()
            if not stmt.startswith("SELECT"):
                raise ValueError("sql_query only allows SELECT statements")

        elif self.kind == ActionKind.CODE_ACT:
            if not self.python_code:
                raise ValueError("code_act requires python_code")

        elif self.kind == ActionKind.OCR_DOCUMENT:
            if not self.pdf_path:
                raise ValueError("ocr_document requires pdf_path")

        elif self.kind == ActionKind.COMPARE_DOC_VS_CLAIM:
            if not self.claim_id:
                raise ValueError("compare_doc_vs_claim requires claim_id")
            if not self.extracted_fields:
                raise ValueError("compare_doc_vs_claim requires extracted_fields")

        elif self.kind == ActionKind.SUBMIT_CASE:
            if not self.case_summary:
                raise ValueError("submit_case requires case_summary")

        return self


# ─── Observation Schema ────────────────────────────────────────────────────────

class FraudHunterObservation(Observation):
    """
    Environment → agent response.

    On reset(): case_brief is populated with the whistleblower dossier.
    On step():  tool_output carries query results or sandbox stdout.
                grader_feedback carries the RLVR scoring rationale.
                evidence_graph shows the accumulated entity/link/contradiction graph.
    """

    case_brief:       Optional[str]            = Field(default=None)
    tool_output:      Optional[str]            = Field(default=None)
    grader_feedback:  Optional[str]            = Field(default=None)
    evidence_graph:   Optional[Dict[str, Any]] = Field(
        default=None, description="Accumulated evidence: entities, links, contradictions"
    )
    step_count:       int  = Field(default=0)
    budget_remaining: int  = Field(default=0)
    difficulty_tier:  int  = Field(default=1, description="RLVE tier 1–5")
    info:             Optional[Dict[str, Any]] = Field(default=None)


# ─── Reward Schedule (RLVR — all numeric constants live here) ─────────────────

# Format gate
FORMAT_GATE_PENALTY       = -10.0   # schema violation → episode ends
COT_MISSING_PENALTY       = -2.0    # no <think> block (soft, non-terminal)
COT_GROUNDED_BONUS        = +1.0    # CoT facts verified against tool output

# Step costs
STEP_DECAY                = -0.1    # mild time pressure per step
DUPLICATE_QUERY_PENALTY   = -5.0    # same query issued twice

# Hallucination / NPI
HALLUCINATED_ENTITY_PENALTY = -50.0  # entity not in database
NPI_EXACT_MATCH_BONUS       = +25.0  # provider NPI perfectly verified
NPI_MISMATCH_PENALTY        = -20.0  # wrong NPI (zero partial credit)

# Per-action task rewards
EXTRACT_ENTITY_REWARD  = +10.0    # correct entity flagged for first time
LINK_SHELL_REWARD      = +50.0    # UBO relationship confirmed
CONTRADICTION_REWARD   = +100.0   # billing anomaly confirmed
CODEACT_BONUS          = +5.0     # per correct SQL result via CodeAct

# Length penalty (phases out after step 20 — forces concision early in training)
LENGTH_PENALTY_RATE    = -0.005   # per excess token beyond 150 in CoT
LENGTH_PENALTY_PHASE_OUT_STEP = 20

# Causal chain bonus (process-based scoring)
PROOF_CHAIN_MULTIPLIER = 1.5      # applied to CASE_WON if full chain confirmed

# Unstructured evidence (OCR on scanned CMS-1500 PDFs)
OCR_RECALL_BONUS       = +20.0    # per field the agent correctly extracts from a PDF
DOC_CLAIM_MATCH_BONUS  = +30.0    # compare_doc_vs_claim finds a document ↔ DB mismatch
PDF_CHAIN_MULTIPLIER   = 1.5      # applied to CONTRADICTION_REWARD when the claim
                                  # cites a PDF-sourced field (multi-modal proof)

# Terminal rewards
CASE_WON_REWARD        = +1000.0
CASE_PARTIAL_REWARD    = +250.0   # won with incomplete evidence
CASE_DISMISSED_REWARD  = 0.0

# Per-typology difficulty multipliers (harder typologies earn more)
TYPOLOGY_MULTIPLIERS: Dict[str, float] = {
    "dead_patient_claim":   1.0,
    "duplicate_bill":       1.0,
    "upcoding":             1.2,
    "unbundling":           1.3,
    "phantom_beneficiary":  1.4,
    "aks_violation":        1.6,
    "off_label_marketing":  1.5,
    "double_billing":       1.2,
    "cost_pricing_fraud":   1.4,
    "product_substitution": 1.5,
    "ppp_fraud":            1.8,
    "foreign_affiliation":  2.0,
}

MAX_EPISODE_STEPS = 60  # increased from 50 to give room for CoT + CodeAct
