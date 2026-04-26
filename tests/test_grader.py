import pytest
from fraud_hunter_env.server.grader import format_gate, grade, GraderOutput, score_cot, validate_npi
from fraud_hunter_env.models import (
    FraudHunterAction, ActionKind, EntityKind,
    FORMAT_GATE_PENALTY, STEP_DECAY, COT_MISSING_PENALTY,
    HALLUCINATED_ENTITY_PENALTY, EXTRACT_ENTITY_REWARD,
    NPI_EXACT_MATCH_BONUS, NPI_MISMATCH_PENALTY,
)
from fraud_hunter_env.server.data_loader import _demo_case


@pytest.fixture
def demo_case():
    """Always use the deterministic in-memory demo case for reproducible tests."""
    return _demo_case()


def test_format_gate_success():
    payload = {"kind": "submit_case", "case_summary": "Test"}
    out = format_gate(payload)
    assert out is None

def test_format_gate_failure():
    payload = {"kind": "submit_case"}
    out = format_gate(payload)
    assert out is not None
    assert out.reward == FORMAT_GATE_PENALTY
    assert out.done is True

def test_cot_missing_penalty():
    cot_r, hits, fb = score_cot(None, 1, {"test"})
    assert cot_r == COT_MISSING_PENALTY
    assert "cot_missing" in hits[0]

def test_cot_grounded_bonus():
    cot_r, hits, fb = score_cot(
        "<think>I see Acme Shell LLC in the registry</think>",
        1,
        {"Acme Shell LLC", "Other Corp"},
    )
    assert cot_r > 0

def test_cot_length_penalty():
    long_cot = " ".join(["word"] * 200)
    cot_r, hits, fb = score_cot(long_cot, 1, set())
    assert cot_r < 0

def test_cot_length_no_penalty_after_phaseout():
    long_cot = " ".join(["word"] * 200)
    cot_r, hits, fb = score_cot(long_cot, 25, set())
    assert all("length" not in h for h in hits)

def test_grade_query_corporate(demo_case):
    action = FraudHunterAction.model_validate({
        "kind": "query_corporate",
        "entity_name": "Acme Shell LLC",
        "think_trace": "<think>Looking up this entity</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1)
    assert "corporate_registry_returned" in out.feedback

def test_npi_exact_match(demo_case):
    """NPI that matches ground truth should get bonus."""
    delta, fb = validate_npi("1234567890", "John Doe MD", demo_case)
    assert delta == NPI_EXACT_MATCH_BONUS
    assert "exact_match" in fb

def test_npi_mismatch(demo_case):
    """Wrong NPI should get penalty."""
    delta, fb = validate_npi("0000000000", "John Doe MD", demo_case)
    assert delta == NPI_MISMATCH_PENALTY
    assert "mismatch" in fb

def test_hallucinated_entity(demo_case):
    action = FraudHunterAction.model_validate({
        "kind": "extract_entity",
        "extracted_name": "Totally Fake Corp",
        "extracted_kind": "corporation",
        "think_trace": "<think>This entity seems suspicious</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1)
    assert out.reward < STEP_DECAY

def test_proof_chain_accumulates(demo_case):
    """Proof trace should grow when extracting a valid ground-truth entity."""
    proof = []
    action = FraudHunterAction.model_validate({
        "kind": "extract_entity",
        "extracted_name": "Acme Shell LLC",
        "extracted_kind": "corporation",
        "think_trace": "<think>Extracting entity</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1, proof_trace=proof)
    assert len(out.proof_trace) > 0
    assert any("entity:" in p for p in out.proof_trace)

def test_typology_multiplier(demo_case):
    """Confirmed contradiction from demo case should earn reward."""
    action = FraudHunterAction.model_validate({
        "kind": "claim_contradiction",
        "evidence_a": "claim:C001",
        "evidence_b": "claim:C002",
        "contradiction_kind": "duplicate_bill",
        "think_trace": "<think>These claims are duplicates</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1)
    assert out.reward > 0
    assert "contradiction_confirmed" in out.feedback

def test_contradiction_fuzzy_prefix_match(demo_case):
    action = FraudHunterAction.model_validate({
        "kind": "claim_contradiction",
        "evidence_a": "BENE_001",
        "evidence_b": "C003",
        "contradiction_kind": "dead_patient_claim",
        "think_trace": "<think>The patient is dead but the claim was billed later.</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1)
    assert out.reward > 0
    assert "contradiction_fuzzy_match" in out.feedback

def test_sql_query_action(demo_case):
    action = FraudHunterAction.model_validate({
        "kind": "sql_query",
        "sql_statement": "SELECT * FROM corporate_registry LIMIT 5",
        "think_trace": "<think>Checking corporate registry</think>",
    })
    out = grade(action, demo_case, set(), set(), set(), False, step_count=1)
    assert out.tool_output is not None
    assert "sql_error" not in out.feedback

def test_duplicate_query_penalty(demo_case):
    action = FraudHunterAction.model_validate({
        "kind": "query_corporate",
        "entity_name": "Acme Shell LLC",
        "think_trace": "<think>Looking up entity</think>",
    })
    grade(action, demo_case, set(), set(), set(), False, step_count=1)
    out2 = grade(action, demo_case, set(), set(), set(), False, step_count=2)
    assert "duplicate_query" in out2.feedback

def test_code_act_rewards_file_reads(demo_case, tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    comms_dir = case_dir / "intercepted_comms"
    comms_dir.mkdir()
    (comms_dir / "email_00.txt").write_text("smoking gun", encoding="utf-8")
    demo_case.db_path = case_dir / "medicare_records.db"

    action = FraudHunterAction.model_validate({
        "kind": "code_act",
        "python_code": "print(open('intercepted_comms/email_00.txt').read())",
        "think_trace": "<think>Read the intercepted email.</think>",
    })
    accessed: list[str] = []
    out = grade(
        action,
        demo_case,
        set(),
        set(),
        set(),
        False,
        step_count=1,
        source_access_callback=accessed.append,
    )
    assert out.reward > STEP_DECAY
    assert "smoking gun" in (out.tool_output or "")
    assert "intercepted_comms/email_00.txt" in accessed
