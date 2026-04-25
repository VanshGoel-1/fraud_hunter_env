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

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from fraud_hunter_env.models import (
    ActionKind, ContradictionKind,
    CASE_DISMISSED_REWARD, CASE_WON_REWARD, CASE_PARTIAL_REWARD,
    CONTRADICTION_REWARD, CODEACT_BONUS, COT_GROUNDED_BONUS,
    COT_MISSING_PENALTY, DOC_CLAIM_MATCH_BONUS, DUPLICATE_QUERY_PENALTY,
    EXTRACT_ENTITY_REWARD, FORMAT_GATE_PENALTY, FraudHunterAction,
    HALLUCINATED_ENTITY_PENALTY, LENGTH_PENALTY_PHASE_OUT_STEP,
    LENGTH_PENALTY_RATE, LINK_SHELL_REWARD, NPI_EXACT_MATCH_BONUS,
    NPI_MISMATCH_PENALTY, OCR_RECALL_BONUS, PDF_CHAIN_MULTIPLIER,
    PROOF_CHAIN_MULTIPLIER, STEP_DECAY, TYPOLOGY_MULTIPLIERS,
)
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

_COT_OPEN_RE  = re.compile(r"<think>", re.IGNORECASE)
_COT_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_WORD_RE      = re.compile(r"\b\w+\b")


@dataclass
class GraderOutput:
    reward: float
    done: bool
    feedback: str
    tool_output: Optional[str] = None
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
    payload = {
        "kind": action.kind.value,
        "entity_name":    action.entity_name,
        "entity_id":      action.entity_id,
        "beneficiary_id": action.beneficiary_id,
        "claim_id":       action.claim_id,
        "sql_statement":  action.sql_statement,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ─── Layer 6: NPI Validator ───────────────────────────────────────────────────

def _validate_npi_luhn(npi: str) -> bool:
    """Simplified NPI Luhn check (prefix 80840 + 9 digits with check digit)."""
    if not npi or len(npi) != 10 or not npi.isdigit():
        return False
    # Standard Luhn on "80840" + NPI[:-1]
    full = "80840" + npi[:-1]
    total = 0
    for i, ch in enumerate(reversed(full)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    return check == int(npi[-1])


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
        stdout, err, rows = execute_code(action.python_code or "", case.conn, case_dir=str(case.db_path.parent))
        if err:
            tool_output = f"SANDBOX_ERROR:\n{err}"
            feedback_parts.append("codeact_error")
        else:
            tool_output = stdout or "(no output)"
            if rows > 0:
                bonus = CODEACT_BONUS * min(rows, 5)  # cap at 5 rows
                reward += bonus
                hits.append(f"codeact_bonus={bonus:.1f}")
            feedback_parts.append(f"codeact_ok({rows}_rows)")

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
        match = (a, b, kind) in gt_ctr or (b, a, kind) in gt_ctr
        already_seen = (a, b) in contradictions or (b, a) in contradictions

        if match and not already_seen:
            # Layer 7: typology multiplier
            multiplier = TYPOLOGY_MULTIPLIERS.get(kind, 1.0)
            typology_reward = CONTRADICTION_REWARD * multiplier
            # Multi-modal proof bonus: if either side cites a PDF, layer
            # PDF_CHAIN_MULTIPLIER on top of the typology multiplier.
            if _evidence_path_referenced(a) or _evidence_path_referenced(b):
                typology_reward *= PDF_CHAIN_MULTIPLIER
                hits.append(f"pdf_chain×{PDF_CHAIN_MULTIPLIER}")
                feedback_parts.append("pdf_chain_proof")
            reward += typology_reward
            hits.append(f"contradiction={typology_reward:.1f}(×{multiplier})")
            feedback_parts.append(f"contradiction_confirmed:{kind}(×{multiplier})")
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
                # Cap the output at 2000 chars to keep the agent's context tight.
                tool_output = (text or "")[:2000]
                feedback_parts.append(f"ocr_ok({len(text or '')}_chars)")

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
            # Bonus: causal chain multiplier
            chain_complete = _check_proof_chain(new_proof_trace)
            base = CASE_WON_REWARD
            if chain_complete:
                base *= PROOF_CHAIN_MULTIPLIER
            reward += base
            hits.append(f"case_won={base:.1f}")
            feedback_parts.append(f"case_won(chain_complete={chain_complete})")
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


def _check_proof_chain(proof_trace: list[str]) -> bool:
    """Returns True if the trace contains at least 1 entity + 1 link + 1 contradiction."""
    has_entity = any(p.startswith("entity:") for p in proof_trace)
    has_link   = any(p.startswith("link:") for p in proof_trace)
    has_contra = any(p.startswith("contradiction:") for p in proof_trace)
    return has_entity and has_link and has_contra


def case_outcome(
    extracted: set[str],
    linked: set[tuple[str, str]],
    contradictions: set[tuple[str, str]],
    case: CaseHandle,
) -> tuple[bool, bool]:
    """Returns (won, partial)."""
    gt_entities = {e["name"].lower() for e in case.ground_truth("entity")}
    gt_links    = {(l["child"].lower(), l["parent"].lower()) for l in case.ground_truth("shell_link")}
    gt_ctr      = {(c["evidence_a"].lower(), c["evidence_b"].lower()) for c in case.ground_truth("contradiction")}

    norm_ctr    = {tuple(sorted(pair)) for pair in contradictions}
    gt_ctr_norm = {tuple(sorted(pair)) for pair in gt_ctr}

    entities_hit = len(gt_entities & extracted)
    links_hit    = len(gt_links & linked)
    contras_hit  = len(gt_ctr_norm & norm_ctr)

    total_gt   = len(gt_entities) + len(gt_links) + len(gt_ctr)
    total_hit  = entities_hit + links_hit + contras_hit

    won     = (
        gt_entities.issubset(extracted)
        and gt_links.issubset(linked)
        and gt_ctr_norm.issubset(norm_ctr)
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
        return 1.0
    return len(queried_tables & gold) / len(gold)
