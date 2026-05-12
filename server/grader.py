"""
RLVR Grader for the Government Fraud Hunter AI environment.

Format-First Hierarchical Curriculum (7 layers):

  Layer 1: JSON Schema Gate
    - Unparseable or schema-invalid action → FORMAT_GATE_PENALTY, episode ends
  Layer 2: CoT Enforcement
    - Missing <think> block → COT_MISSING_PENALTY (soft, non-terminal)
    - Unclosed <think> → harder penalty
  Layer 3: CoT Grounding Verifier
    - Mentions of real entities in CoT → COT_GROUNDED_BONUS
    - Hallucinated entity names in CoT → incremental penalty
  Layer 4: Length Reward Schedule
    - Steps ≤ 20: apply LENGTH_PENALTY_RATE per excess CoT token
    - Steps > 20: disable (model has learned concision)
  Layer 5: Duplicate Query Detection
    - Exact repeat of a prior query → DUPLICATE_QUERY_PENALTY
  Layer 6: NPI Strict Validation
    - Provider extraction: npi_code must exactly match ground truth
    - No partial credit; mismatch → NPI_MISMATCH_PENALTY
  Layer 7: Per-Typology Reward Matrix
    - Each confirmed contradiction earns CONTRADICTION_REWARD × typology multiplier
  Bonus: Process-Based Causal Chain Scoring
    - entity → shell_link → contradiction = complete proof chain → multiplier on terminal reward
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fraud_hunter_env.models import (
    ActionKind, ContradictionKind,
    CASE_DISMISSED_REWARD, CASE_WON_REWARD, CASE_PARTIAL_REWARD,
    CONTRADICTION_REWARD, CODEACT_BONUS, COT_GROUNDED_BONUS,
    COT_MISSING_PENALTY, DOC_CLAIM_MATCH_BONUS, DUPLICATE_QUERY_PENALTY,
    EXTRACT_ENTITY_REWARD, FORMAT_GATE_PENALTY, FraudHunterAction,
    HALLUCINATED_ENTITY_PENALTY, HALLUCINATED_LINK_PENALTY,
    LENGTH_PENALTY_PHASE_OUT_STEP,
    LENGTH_PENALTY_RATE, LINK_SHELL_REWARD, NPI_EXACT_MATCH_BONUS,
    NPI_MISMATCH_PENALTY, OCR_RECALL_BONUS, PDF_CHAIN_MULTIPLIER,
    PROOF_CHAIN_MULTIPLIER, STEP_DECAY, TYPOLOGY_MULTIPLIERS,
)
from fraud_hunter_env.npi_utils import validate_npi_luhn
from fraud_hunter_env.schema import GT_KIND_SOURCES, TYPOLOGY_SOURCES

from pathlib import Path

from .data_loader import CaseHandle
from .sandbox import execute_code, execute_sql


# ─── OCR helpers ──────────────────────────────────────────────────────────────

def _ocr_pdf(pdf_abs_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract text from a scanned PDF. Returns (text, error)."""
    try:
        import pdfplumber
    except ImportError:
        return None, "pdfplumber_not_installed"

    if not pdf_abs_path.exists():
        return None, f"pdf_not_found:{pdf_abs_path.name}"

    try:
        with pdfplumber.open(str(pdf_abs_path)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        return text, None
    except Exception as exc:
        return None, f"ocr_failed:{exc}"


def _normalize_field(value: Any) -> str:
    """Lower-case, strip, drop $/% so '$350.00' and '350.0' compare equal."""
    if value is None:
        return ""
    s = str(value).strip().lower().replace("$", "").replace(",", "").replace("%", "")
    try:
        return f"{float(s):.4f}"
    except ValueError:
        return s


def _evidence_path_referenced(text: str) -> bool:
    """Detect a PDF / scanned_claims reference inside an evidence string."""
    if not text:
        return False
    t = text.lower()
    return ("scanned_claims/" in t) or t.startswith("doc_") or t.endswith(".pdf")


_BENE_ID_RE = re.compile(r"\bbene[_-]?[0-9]+\b", re.IGNORECASE)
_CLAIM_ID_RE = re.compile(r"\b(?:c|clm|claim)[_:-]?[a-z0-9_]+\b", re.IGNORECASE)


def _normalize_ocr_digits(text: str) -> str:
    return (
        text.upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
    )


def _canonicalize_evidence_token(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("beneficiary:") or raw.startswith("claim:"):
        return raw
    if raw.startswith("provider_npi:"):
        prefix, _, suffix = raw.partition(":")
        return f"{prefix}:{_normalize_ocr_digits(suffix).lower()}"

    bene_match = _BENE_ID_RE.search(raw)
    if bene_match:
        token = bene_match.group(0).upper().replace("-", "_")
        return f"beneficiary:{token}".lower()

    if raw.startswith("c_") or raw.startswith("claim_") or raw.startswith("clm_"):
        return f"claim:{raw.replace('claim_', 'c_')}".lower()

    claim_match = _CLAIM_ID_RE.search(raw)
    if claim_match:
        token = claim_match.group(0).replace("claim:", "")
        token = token.replace("claim_", "c_", 1)
        token = token.replace("clm_", "c_", 1)
        return f"claim:{token}".lower()

    return raw


def _contradiction_match_type(
    evidence_a: str,
    evidence_b: str,
    kind: str,
    ground_truth: set[tuple[str, str, str]],
) -> str:
    raw_pair = (evidence_a.lower(), evidence_b.lower(), kind)
    if raw_pair in ground_truth or (raw_pair[1], raw_pair[0], kind) in ground_truth:
        return "exact"

    canonical_a = _canonicalize_evidence_token(evidence_a)
    canonical_b = _canonicalize_evidence_token(evidence_b)
    canonical_pair = (canonical_a, canonical_b, kind)
    if canonical_pair in ground_truth or (canonical_pair[1], canonical_pair[0], kind) in ground_truth:
        return "fuzzy"
    return "none"

_COT_OPEN_RE  = re.compile(r"<think>", re.IGNORECASE)
_COT_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_WORD_RE      = re.compile(r"\b\w+\b")


# OCR output cap — tier-scaled. Tier-1 cases have small single-page CMS-1500
# forms; Tier-5 cases ship multi-page degraded scans where the contradiction
# fields can live past the first 2k chars. Defaults to 4000 for unknown tiers.
_OCR_CAP_BY_TIER: dict[int, int] = {1: 1500, 2: 2000, 3: 3000, 4: 4000, 5: 4000}
_OCR_CAP_DEFAULT: int = 4000


def _ocr_cap_for(tier: int) -> int:
    return _OCR_CAP_BY_TIER.get(tier, _OCR_CAP_DEFAULT)


@dataclass
class GraderOutput:
    reward: float
    done: bool
    feedback: str
    tool_output: Optional[str] = None
    base64_document: Optional[str] = None
    hits: list[str] = field(default_factory=list)       # human-readable reward ledger
    proof_trace: list[str] = field(default_factory=list) # for causal chain scoring

    def merge_tool(self, text: str) -> "GraderOutput":
        self.tool_output = text
        return self


# ─── Layer 1: Format Gate ─────────────────────────────────────────────────────

def format_gate(action_payload: dict[str, Any]) -> Optional[GraderOutput]:
    """Returns GraderOutput on schema failure (episode-terminating), None on success."""
    try:
        FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return GraderOutput(
            reward=FORMAT_GATE_PENALTY,
            done=True,
            feedback=f"schema_violation: {exc}",
            hits=[f"format_gate={FORMAT_GATE_PENALTY}"],
        )
    return None


# ─── Layer 2 & 3: CoT Checker ─────────────────────────────────────────────────

def score_cot(
    think_trace: Optional[str],
    step_count: int,
    real_names: set[str],
) -> tuple[float, list[str], list[str]]:
    """
    Returns (cot_reward, cot_hits, cot_feedback_parts).
    Checks presence, closure, length penalty, and entity grounding.
    """
    cot_reward = 0.0
    hits: list[str] = []
    feedback: list[str] = []

    if not think_trace:
        cot_reward += COT_MISSING_PENALTY
        hits.append(f"cot_missing={COT_MISSING_PENALTY}")
        feedback.append("no_cot_trace")
        return cot_reward, hits, feedback

    has_open  = bool(_COT_OPEN_RE.search(think_trace))
    has_close = bool(_COT_CLOSE_RE.search(think_trace))

    if has_open and not has_close:
        cot_reward -= 5.0
        hits.append("cot_unclosed=-5.0")
        feedback.append("cot_unclosed")

    # Layer 4: Length penalty (phases out after step 20)
    if step_count <= LENGTH_PENALTY_PHASE_OUT_STEP:
        words = _WORD_RE.findall(think_trace)
        excess = max(0, len(words) - 150)
        length_pen = LENGTH_PENALTY_RATE * excess
        if length_pen < 0:
            cot_reward += length_pen
            hits.append(f"length_penalty={length_pen:.3f}")
            feedback.append(f"cot_too_long({len(words)}_words)")

    # Layer 3: Grounding — reward if CoT mentions real entity names
    cot_lower = think_trace.lower()
    grounded = sum(1 for n in real_names if n.lower() in cot_lower)
    if grounded > 0:
        bonus = COT_GROUNDED_BONUS * min(grounded, 3)  # cap at 3x bonus
        cot_reward += bonus
        hits.append(f"cot_grounded={bonus:.1f}")
        feedback.append(f"cot_grounded({grounded}_entities)")

    return cot_reward, hits, feedback


# ─── Layer 5: Duplicate Detection ─────────────────────────────────────────────

def query_hash(action: FraudHunterAction) -> str:
    """Stable identity for an information-gathering action. Used to penalise
    repeated queries. Includes ``python_code`` and ``pdf_path`` so duplicate
    CodeAct probes and OCR re-reads are caught (they previously slipped through
    because only the structured-query fields were hashed).
    """
    payload = {
        "kind": action.kind.value,
        "entity_name":    action.entity_name,
        "entity_id":      action.entity_id,
        "beneficiary_id": action.beneficiary_id,
        "claim_id":       action.claim_id,
        "sql_statement":  action.sql_statement,
        "python_code":    action.python_code,
        "pdf_path":       action.pdf_path,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ─── Layer 6: NPI Validator ───────────────────────────────────────────────────


def validate_npi(
    npi_code: Optional[str],
    extracted_name: str,
    case: CaseHandle,
) -> tuple[float, str]:
    """
    Returns (reward_delta, feedback_str).
    Looks up the ground-truth NPI for the provider name and compares exactly.
    Zero partial credit.
    """
    if not npi_code:
        return 0.0, "npi_not_provided"

    if not validate_npi_luhn(npi_code):
        return NPI_MISMATCH_PENALTY, f"npi_luhn_fail:{npi_code}"

    # Look up ground-truth NPI from corporate_registry
    cur = case.conn.execute(
        "SELECT npi_code FROM corporate_registry WHERE entity_name = ? COLLATE NOCASE LIMIT 1",
        (extracted_name,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return 0.0, "npi_provider_not_in_registry"

    gt_npi = str(row[0])
    if npi_code == gt_npi:
        return NPI_EXACT_MATCH_BONUS, f"npi_exact_match={gt_npi}"
    else:
        return NPI_MISMATCH_PENALTY, f"npi_mismatch(got={npi_code},expected={gt_npi})"


# ─── Main Grader ──────────────────────────────────────────────────────────────

def grade(
    action: FraudHunterAction,
    case: CaseHandle,
    extracted: set[str],
    linked: set[tuple[str, str]],
    contradictions: set[tuple[str, str]],
    submitted: bool,
    step_count: int = 0,
    proof_trace: Optional[list[str]] = None,
    source_access_callback: Optional[Callable[[str], None]] = None,
    sql_trace_callback: Optional[Callable[[str], None]] = None,
) -> GraderOutput:
    """
    Score a single step. Caller owns state mutation; grader only reads.
    Returns GraderOutput with reward, done, feedback, tool_output, hits.
    """
    if proof_trace is None:
        proof_trace = []

    if submitted:
        return GraderOutput(
            reward=0.0, done=True,
            feedback="episode_already_submitted",
            hits=["post_submit_noop"],
        )

    reward = STEP_DECAY
    hits: list[str] = [f"step_decay={STEP_DECAY}"]
    feedback_parts: list[str] = []
    tool_output: Optional[str] = None
    done = False
    new_proof_trace = list(proof_trace)

    # Get all real entity names for CoT grounding
    real_names = case.all_entity_names()

    # Layer 2 & 3: CoT scoring
    cot_r, cot_hits, cot_fb = score_cot(action.think_trace, step_count, real_names)
    reward += cot_r
    hits.extend(cot_hits)
    feedback_parts.extend(cot_fb)

    # Layer 5: Duplicate detection for query-type actions
    if action.kind in (
        ActionKind.QUERY_CORPORATE,
        ActionKind.QUERY_MEDICARE,
        ActionKind.SQL_QUERY,
        ActionKind.CODE_ACT,
        ActionKind.OCR_DOCUMENT,
    ):
        qh = query_hash(action)
        if qh in case.seen_queries:
            reward += DUPLICATE_QUERY_PENALTY
            hits.append(f"duplicate={DUPLICATE_QUERY_PENALTY}")
            feedback_parts.append("duplicate_query")
        case.seen_queries.add(qh)

    # ── Action Dispatch ────────────────────────────────────────────────────────

    if action.kind == ActionKind.QUERY_CORPORATE:
        tool_output = case.query_corporate(action.entity_name, action.entity_id)
        feedback_parts.append("corporate_registry_returned")

    elif action.kind == ActionKind.QUERY_MEDICARE:
        tool_output = case.query_medicare(action.beneficiary_id, action.claim_id)
        feedback_parts.append("medicare_returned")

    elif action.kind == ActionKind.SQL_QUERY:
        output, err, rows = execute_sql(action.sql_statement or "", case.conn)
        if err:
            tool_output = f"SQL_ERROR: {err}"
            feedback_parts.append("sql_error")
        else:
            tool_output = output
            feedback_parts.append(f"sql_ok({rows}_rows)")
            # Bonus for productive SQL (returned rows)
            if rows > 0:
                reward += min(rows * 0.5, 5.0)  # cap at +5.0
                hits.append(f"sql_rows_bonus={min(rows * 0.5, 5.0):.1f}")

    elif action.kind == ActionKind.CODE_ACT:
        stdout, err, stats = execute_code(
            action.python_code or "",
            case.conn,
            case_dir=str(case.db_path.parent),
            on_access=source_access_callback,
            on_sql=sql_trace_callback,
        )
        if err:
            tool_output = f"SANDBOX_ERROR:\n{err}"
            feedback_parts.append("codeact_error")
        else:
            tool_output = stdout or "(no output)"
            rows = int(stats.get("rows_returned", 0))
            files_read = int(stats.get("files_read", 0))
            directories_listed = int(stats.get("directories_listed", 0))
            bonus = 0.0
            if rows > 0:
                bonus += CODEACT_BONUS * min(rows, 5)
            if files_read > 0:
                bonus += min(files_read, 3) * 2.5
            if directories_listed > 0:
                bonus += min(directories_listed, 2) * 1.0
            if bonus > 0:
                reward += bonus
                hits.append(f"codeact_bonus={bonus:.1f}")
            feedback_parts.append(
                f"codeact_ok(rows={rows},files={files_read},dirs={directories_listed})"
            )

    elif action.kind == ActionKind.EXTRACT_ENTITY:
        name = (action.extracted_name or "").strip()
        kind = action.extracted_kind.value if action.extracted_kind else ""
        gt_entities = {
            (e["name"].lower(), e["kind"])
            for e in case.ground_truth("entity")
            if "name" in e and "kind" in e
        }

        if not name or name.lower() not in {n.lower() for n in real_names}:
            reward += HALLUCINATED_ENTITY_PENALTY
            hits.append(f"hallucination={HALLUCINATED_ENTITY_PENALTY}")
            feedback_parts.append(f"hallucinated_entity:{name!r}")
        elif (name.lower(), kind) in gt_entities and name.lower() not in extracted:
            reward += EXTRACT_ENTITY_REWARD
            hits.append(f"extract={EXTRACT_ENTITY_REWARD}")
            feedback_parts.append(f"extracted:{name!r}")
            new_proof_trace.append(f"entity:{name.lower()}")

            # Layer 6: NPI validation for providers
            if kind == "provider":
                npi_delta, npi_fb = validate_npi(action.npi_code, name, case)
                reward += npi_delta
                hits.append(f"npi={npi_delta:.1f}")
                feedback_parts.append(npi_fb)
        else:
            feedback_parts.append(f"already_extracted_or_off_target:{name!r}")

    elif action.kind == ActionKind.LINK_SHELL:
        child  = (action.child_entity  or "").lower()
        parent = (action.parent_entity or "").lower()
        gt_links = {
            (l["child"].lower(), l["parent"].lower())
            for l in case.ground_truth("shell_link")
        }
        if (child, parent) in gt_links and (child, parent) not in linked:
            reward += LINK_SHELL_REWARD
            hits.append(f"shell_link={LINK_SHELL_REWARD}")
            feedback_parts.append(f"shell_link_confirmed:{child}→{parent}")
            new_proof_trace.append(f"link:{child}→{parent}")
        elif (child, parent) in linked:
            feedback_parts.append(f"shell_link_already_recorded:{child}→{parent}")
        else:
            # Penalise asserting a link between entities that don't exist or
            # have no DB-grounded relationship. Mirrors HALLUCINATED_ENTITY_PENALTY
            # for extract_entity so shotgun-guessing links isn't free.
            real = case.all_entity_names()
            real_lower = {n.lower() for n in real}
            both_in_db = (child in real_lower) and (parent in real_lower)
            if not both_in_db:
                reward += HALLUCINATED_LINK_PENALTY
                hits.append(f"hallucinated_link={HALLUCINATED_LINK_PENALTY}")
                feedback_parts.append(f"hallucinated_link:{child}→{parent}")
            else:
                feedback_parts.append(f"shell_link_unconfirmed:{child}→{parent}")

    elif action.kind == ActionKind.CLAIM_CONTRADICTION:
        a    = (action.evidence_a or "").lower()
        b    = (action.evidence_b or "").lower()
        kind = action.contradiction_kind.value if action.contradiction_kind else ""
        gt_ctr = {
            (c["evidence_a"].lower(), c["evidence_b"].lower(), c["kind"])
            for c in case.ground_truth("contradiction")
        }
        match_type = _contradiction_match_type(a, b, kind, gt_ctr)
        already_seen = (a, b) in contradictions or (b, a) in contradictions

        if match_type != "none" and not already_seen:
            # Layer 7: typology multiplier
            multiplier = TYPOLOGY_MULTIPLIERS.get(kind, 1.0)
            typology_reward = CONTRADICTION_REWARD * multiplier
            if match_type == "fuzzy":
                typology_reward *= 0.4
            # Multi-modal proof bonus: if either side cites a PDF, layer
            # PDF_CHAIN_MULTIPLIER on top of the typology multiplier.
            if _evidence_path_referenced(a) or _evidence_path_referenced(b):
                typology_reward *= PDF_CHAIN_MULTIPLIER
                hits.append(f"pdf_chain×{PDF_CHAIN_MULTIPLIER}")
                feedback_parts.append("pdf_chain_proof")
            reward += typology_reward
            if match_type == "exact":
                hits.append(f"contradiction={typology_reward:.1f}(×{multiplier})")
                feedback_parts.append(f"contradiction_confirmed:{kind}(×{multiplier})")
            else:
                hits.append(f"contradiction_fuzzy={typology_reward:.1f}(×{multiplier})")
                feedback_parts.append(f"contradiction_fuzzy_match:{kind}(×{multiplier})")
            new_proof_trace.append(f"contradiction:{kind}:{a}↔{b}")
        else:
            feedback_parts.append(f"contradiction_unconfirmed:{kind}")

    elif action.kind == ActionKind.OCR_DOCUMENT:
        # Resolve the PDF path relative to the case directory.
        rel_pdf = (action.pdf_path or "").lstrip("/").lstrip("\\")
        case_root = case.db_path.parent
        pdf_abs = (case_root / rel_pdf).resolve()
        # Path-traversal guard: pdf must live under the case dir.
        try:
            pdf_abs.relative_to(case_root.resolve())
        except ValueError:
            tool_output = f"OCR_ERROR: path_outside_case:{rel_pdf!r}"
            feedback_parts.append("ocr_path_violation")
        else:
            text, err = _ocr_pdf(pdf_abs)
            if err:
                tool_output = f"OCR_ERROR: {err}"
                feedback_parts.append("ocr_error")
            else:
                # Cap the output to keep the agent's context tight; tier-scaled
                # so harder cases (multi-page PDFs) get more headroom.
                tool_output = (text or "")[:_ocr_cap_for(case.tier)]
                encoded = base64.b64encode(pdf_abs.read_bytes()).decode("ascii")
                feedback_parts.append(f"ocr_ok({len(text or '')}_chars)")
                return GraderOutput(
                    reward=round(reward, 4),
                    done=done,
                    feedback="; ".join(feedback_parts) or "noop",
                    tool_output=tool_output,
                    base64_document=encoded,
                    hits=hits,
                    proof_trace=new_proof_trace,
                )

    elif action.kind == ActionKind.COMPARE_DOC_VS_CLAIM:
        claim_id = action.claim_id or ""
        extracted_fields = action.extracted_fields or {}
        # Pull expected (printed-on-PDF) fields from evidence_documents.
        try:
            cur = case.conn.execute(
                "SELECT expected_fields_json FROM evidence_documents WHERE claim_id = ?",
                (claim_id,),
            )
            row = cur.fetchone()
        except Exception as exc:
            row = None
            feedback_parts.append(f"compare_lookup_error:{exc}")

        if not row or not row[0]:
            tool_output = f"no_evidence_doc_for_claim:{claim_id!r}"
            feedback_parts.append("compare_doc_missing")
        else:
            try:
                expected: dict = json.loads(row[0])
            except json.JSONDecodeError:
                expected = {}

            # Pull the live DB-of-record value for the same claim.
            db_row: dict[str, Any] = {}
            try:
                cc = case.conn.execute(
                    "SELECT CLM_ID, PRF_PHYSN_NPI, HCPCS_CD, LINE_NCH_PMT_AMT, "
                    "CLM_FROM_DT FROM carrier_claims WHERE CLM_ID = ?",
                    (claim_id,),
                )
                r = cc.fetchone()
                if r:
                    db_row = {
                        "claim_id": r[0], "npi": r[1], "hcpcs_code": r[2],
                        "amount": r[3], "service_date": r[4],
                    }
            except Exception:
                pass

            ocr_correct = 0
            doc_db_mismatch = 0
            mismatched_keys: list[str] = []
            for k, v in extracted_fields.items():
                exp_v = expected.get(k)
                db_v = db_row.get(k)
                if exp_v is None:
                    continue
                if _normalize_field(v) == _normalize_field(exp_v):
                    ocr_correct += 1
                    if db_v is not None and _normalize_field(exp_v) != _normalize_field(db_v):
                        doc_db_mismatch += 1
                        mismatched_keys.append(k)

            ocr_bonus = OCR_RECALL_BONUS * min(ocr_correct, 5)
            mismatch_bonus = DOC_CLAIM_MATCH_BONUS * doc_db_mismatch
            reward += ocr_bonus + mismatch_bonus
            if ocr_bonus > 0:
                hits.append(f"ocr_recall={ocr_bonus:.1f}({ocr_correct}_fields)")
            if mismatch_bonus > 0:
                hits.append(f"doc_db_mismatch={mismatch_bonus:.1f}")
                # Multi-modal proof: a doc-vs-DB mismatch is itself a proof step.
                new_proof_trace.append(
                    f"doc_db_mismatch:{claim_id}:{','.join(mismatched_keys)}"
                )
            tool_output = (
                f"ocr_correct={ocr_correct} mismatches={doc_db_mismatch} "
                f"keys={mismatched_keys}"
            )
            feedback_parts.append(
                f"compare_doc_vs_claim(correct={ocr_correct},mismatch={doc_db_mismatch})"
            )

    elif action.kind == ActionKind.SUBMIT_CASE:
        done = True
        won, partial = case_outcome(extracted, linked, contradictions, case)
        if won:
            # Bonus: causal chain multiplier (scaled by tier so a Tier-5 win
            # actually requires a Tier-5-sized chain, not just one of each).
            chain_complete = _check_proof_chain(new_proof_trace, case.tier)
            base = CASE_WON_REWARD
            if chain_complete:
                base *= PROOF_CHAIN_MULTIPLIER
            reward += base
            hits.append(f"case_won={base:.1f}")
            feedback_parts.append(f"case_won(chain_complete={chain_complete},tier={case.tier})")
        elif partial:
            reward += CASE_PARTIAL_REWARD
            hits.append(f"case_partial={CASE_PARTIAL_REWARD}")
            feedback_parts.append("case_partial")
        else:
            reward += CASE_DISMISSED_REWARD
            hits.append(f"case_dismissed={CASE_DISMISSED_REWARD}")
            feedback_parts.append("case_dismissed")

    return GraderOutput(
        reward=round(reward, 4),
        done=done,
        feedback="; ".join(feedback_parts) or "noop",
        tool_output=tool_output,
        hits=hits,
        proof_trace=new_proof_trace,
    )


# Win-threshold schedule: fraction of each ground-truth bucket the agent must
# cover for a CASE_WON outcome. Strict at low tiers (where the case is small
# enough to fully solve in 60 steps) and relaxed at high tiers (Tier-5 cases
# have ~26 GT items — requiring 100% coverage made CASE_WON practically
# unreachable, which dead-ended the policy gradient at high tiers).
_WIN_THRESHOLD_BY_TIER: dict[int, float] = {
    1: 1.00,
    2: 0.90,
    3: 0.80,
    4: 0.70,
    5: 0.60,
}


def _required_chain_count(tier: int) -> int:
    """Per-bucket count the proof trace must cover to earn the multiplier.

    Scales with tier so that one entity + one link + one contradiction does
    not buy a 1.5× bonus on a 26-row Tier-5 case.
    """
    return max(1, min(tier, 5))


def _check_proof_chain(proof_trace: list[str], tier: int = 1) -> bool:
    """Returns True if the trace covers at least ``_required_chain_count(tier)``
    of each of {entity, link, contradiction}.
    """
    required = _required_chain_count(tier)
    n_entity = sum(1 for p in proof_trace if p.startswith("entity:"))
    n_link   = sum(1 for p in proof_trace if p.startswith("link:"))
    n_contra = sum(1 for p in proof_trace if p.startswith("contradiction:"))
    return n_entity >= required and n_link >= required and n_contra >= required


def case_outcome(
    extracted: set[str],
    linked: set[tuple[str, str]],
    contradictions: set[tuple[str, str]],
    case: CaseHandle,
) -> tuple[bool, bool]:
    """Returns (won, partial)."""
    gt_entities = {e["name"].lower() for e in case.ground_truth("entity")}
    gt_links    = {(l["child"].lower(), l["parent"].lower()) for l in case.ground_truth("shell_link")}
    gt_ctr      = {
        (
            _canonicalize_evidence_token(c["evidence_a"]),
            _canonicalize_evidence_token(c["evidence_b"]),
        )
        for c in case.ground_truth("contradiction")
    }

    norm_ctr    = {
        tuple(sorted((_canonicalize_evidence_token(pair[0]), _canonicalize_evidence_token(pair[1]))))
        for pair in contradictions
    }
    gt_ctr_norm = {tuple(sorted(pair)) for pair in gt_ctr}

    entities_hit = len(gt_entities & extracted)
    links_hit    = len(gt_links & linked)
    contras_hit  = len(gt_ctr_norm & norm_ctr)

    total_gt   = len(gt_entities) + len(gt_links) + len(gt_ctr)
    total_hit  = entities_hit + links_hit + contras_hit

    # Tier-scaled win condition: each non-empty bucket must clear the tier's
    # coverage threshold. Empty buckets are vacuously satisfied so cases with
    # only contradictions (no shell_link planted) can still be won.
    threshold = _WIN_THRESHOLD_BY_TIER.get(case.tier, 0.6)

    def _bucket_ok(hit: int, total: int) -> bool:
        return total == 0 or (hit / total) >= threshold

    won = (
        _bucket_ok(entities_hit, len(gt_entities))
        and _bucket_ok(links_hit, len(gt_links))
        and _bucket_ok(contras_hit, len(gt_ctr_norm))
    )
    partial = not won and total_gt > 0 and total_hit / total_gt >= 0.5

    return won, partial


def compute_agentic_recall(
    queried_tables: set[str],
    case: CaseHandle,
) -> float:
    """
    Agentic Recall = tables/evidence queried / gold required sources.

    Multi-modal: includes the filesystem evidence directories because the AKS
    golden thread lives half in SQL (PDE spike, general_ledger, referral_payments)
    and half in the filesystem (intercepted_comms smoking gun, scanned_claims PDFs).
    """
    gt_kinds = {row[0] for row in case.conn.execute("SELECT DISTINCT kind FROM ground_truth")}
    gold: set[str] = set()
    for kind in gt_kinds:
        gold.update(GT_KIND_SOURCES.get(kind, frozenset()))

    # Add typology-specific sources by reading the contradiction payloads.
    try:
        rows = case.conn.execute(
            "SELECT payload_json FROM ground_truth WHERE kind = 'contradiction'"
        ).fetchall()
        for (raw,) in rows:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = payload.get("kind", "")
            gold.update(TYPOLOGY_SOURCES.get(kind, frozenset()))
    except Exception:
        pass

    if not gold:
        return 0.0
    return len(queried_tables & gold) / len(gold)
