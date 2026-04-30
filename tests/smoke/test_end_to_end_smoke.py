from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_hunter_env.server.app import app
from fraud_hunter_env.server.http_contract import post_testclient_json, step_testclient_json


@pytest.mark.smoke
def test_case_brief_documents_contradiction_contract_and_code_act_tooling():
    client = TestClient(app)

    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": None, "seed_max": None})
    reset_payload = post_testclient_json(client, "/reset")

    case_brief = (((reset_payload or {}).get("observation") or {}).get("case_brief") or "")

    assert "beneficiary:[DESYNPUF_ID]" in case_brief
    assert "claim:[CLM_ID]" in case_brief
    assert "Use code_act to inspect intercepted_comms/*.txt" in case_brief


@pytest.mark.smoke
def test_http_smoke_for_critical_actions_and_multimodal_payload():
    client = TestClient(app)

    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": 8001, "seed_max": 10000})
    post_testclient_json(client, "/reset")

    pdf_path = "scanned_claims/doc_claim.pdf"

    actions = [
        {
            "kind": "link_shell",
            "child_entity": "Acme Shell LLC",
            "parent_entity": "Starpoint Holdings Ltd",
            "think_trace": "<think>Validate HTTP serialization for link_shell.</think>",
        },
        {
            "kind": "claim_contradiction",
            "evidence_a": "beneficiary:BENE_001",
            "evidence_b": "claim:C003",
            "contradiction_kind": "dead_patient_claim",
            "think_trace": "<think>Validate HTTP serialization for claim_contradiction.</think>",
        },
        {
            "kind": "code_act",
            "python_code": "files = listdir('intercepted_comms')\nprint(files[:1])",
            "think_trace": "<think>Validate HTTP serialization for code_act.</think>",
        },
        {
            "kind": "ocr_document",
            "pdf_path": pdf_path,
            "think_trace": "<think>Validate HTTP serialization for ocr_document.</think>",
        },
        {
            "kind": "compare_doc_vs_claim",
            "claim_id": "C003",
            "extracted_fields": {
                "claim_id": "C003",
                "amount": 480.0,
                "hcpcs_code": "99214",
            },
            "think_trace": "<think>Validate HTTP serialization for compare_doc_vs_claim.</think>",
        },
    ]

    ocr_payload = None
    for action in actions:
        payload = step_testclient_json(client, action)
        observation = payload["observation"]
        assert "base64_document" in observation
        assert "grader_feedback" in observation
        assert isinstance(payload["reward"], (int, float))
        if action["kind"] == "ocr_document":
            ocr_payload = payload

    assert ocr_payload is not None
    base64_document = (((ocr_payload or {}).get("observation") or {}).get("base64_document"))
    assert base64_document is None or isinstance(base64_document, str)


@pytest.mark.smoke
def test_agentic_recall_is_not_gameable_by_sql_string_literals():
    client = TestClient(app)

    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": None, "seed_max": None})
    post_testclient_json(client, "/reset")

    payload = step_testclient_json(
        client,
        {
            "kind": "sql_query",
            "sql_statement": "SELECT 'corporate_registry', 'beneficiary_summary'",
            "think_trace": "<think>Literal strings should not count as table access.</think>",
        },
    )

    info = (((payload or {}).get("observation") or {}).get("info") or {})
    assert info.get("agentic_recall") == 0.0


@pytest.mark.smoke
def test_seed_split_respects_train_and_eval_ranges_over_http():
    client = TestClient(app)

    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": 0, "seed_max": 8000})
    train_range = client.get("/fraud_hunter/seed_range")
    train_range.raise_for_status()
    assert train_range.json()["seed_min"] == 0
    assert train_range.json()["seed_max"] == 8000

    train_reset = post_testclient_json(client, "/reset")
    train_info = (((train_reset or {}).get("observation") or {}).get("info") or {})
    assert "case_id" in train_info

    # Scoped ranges are immutable once pinned unless explicitly cleared.
    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": None, "seed_max": None})
    post_testclient_json(client, "/fraud_hunter/seed_range", {"seed_min": 8001, "seed_max": 10000})
    eval_range = client.get("/fraud_hunter/seed_range")
    eval_range.raise_for_status()
    assert eval_range.json()["seed_min"] == 8001
    assert eval_range.json()["seed_max"] == 10000

    eval_reset = post_testclient_json(client, "/reset")
    eval_info = (((eval_reset or {}).get("observation") or {}).get("info") or {})
    assert "case_id" in eval_info
