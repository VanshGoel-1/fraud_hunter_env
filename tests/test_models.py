import pytest
from pydantic import ValidationError

from fraud_hunter_env.models import FraudHunterAction, ActionKind, EntityKind, ContradictionKind


def test_valid_corporate_query():
    action = FraudHunterAction.model_validate({
        "kind": "query_corporate",
        "entity_name": "Acme Shell LLC"
    })
    assert action.kind == ActionKind.QUERY_CORPORATE

def test_missing_fields_format_gate():
    with pytest.raises(ValidationError):
        FraudHunterAction.model_validate({
            "kind": "extract_entity"
        })

def test_valid_extract_entity_provider_requires_npi():
    """Provider extraction MUST include npi_code."""
    with pytest.raises(ValidationError, match="npi_code"):
        FraudHunterAction.model_validate({
            "kind": "extract_entity",
            "extracted_name": "Dr. Fraud",
            "extracted_kind": "provider",
        })

def test_valid_extract_entity_provider_with_npi():
    action = FraudHunterAction.model_validate({
        "kind": "extract_entity",
        "extracted_name": "Dr. Fraud",
        "extracted_kind": "provider",
        "npi_code": "1234567890",
    })
    assert action.npi_code == "1234567890"

def test_valid_submit():
    action = FraudHunterAction.model_validate({
        "kind": "submit_case",
        "case_summary": "We found fraud.",
        "typologies": ["dead_patient_claim", "duplicate_bill"],
    })
    assert action.kind == ActionKind.SUBMIT_CASE
    assert action.typologies == ["dead_patient_claim", "duplicate_bill"]

def test_sql_query_must_be_select():
    with pytest.raises(ValidationError, match="SELECT"):
        FraudHunterAction.model_validate({
            "kind": "sql_query",
            "sql_statement": "DROP TABLE providers",
        })

def test_valid_sql_query():
    action = FraudHunterAction.model_validate({
        "kind": "sql_query",
        "sql_statement": "SELECT * FROM providers",
    })
    assert action.kind == ActionKind.SQL_QUERY

def test_code_act_requires_code():
    with pytest.raises(ValidationError):
        FraudHunterAction.model_validate({
            "kind": "code_act",
        })

def test_valid_code_act():
    action = FraudHunterAction.model_validate({
        "kind": "code_act",
        "python_code": "result = pd.read_sql('SELECT * FROM providers', conn)",
    })
    assert action.kind == ActionKind.CODE_ACT

def test_contradiction_kinds_expanded():
    """All 12 contradiction kinds should be valid."""
    for kind in ContradictionKind:
        action = FraudHunterAction.model_validate({
            "kind": "claim_contradiction",
            "evidence_a": "claim:C001",
            "evidence_b": "claim:C002",
            "contradiction_kind": kind.value,
        })
        assert action.contradiction_kind == kind

def test_think_trace_optional():
    action = FraudHunterAction.model_validate({
        "kind": "query_corporate",
        "entity_name": "Test Corp",
        "think_trace": "<think>Looking up this entity</think>",
    })
    assert action.think_trace is not None
