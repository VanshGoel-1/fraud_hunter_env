from __future__ import annotations

import os

from fastapi.testclient import TestClient

from fraud_hunter_env.server.app import app


def _first_pdf_in_case(reset_obs: dict) -> str:
    """Return the first real PDF path from the reset observation's case_dir."""
    case_dir = ((reset_obs.get("observation") or {}).get("info") or {}).get("case_dir", "")
    if case_dir:
        sc_dir = os.path.join(case_dir, "scanned_claims")
        try:
            pdfs = sorted(f for f in os.listdir(sc_dir) if f.endswith(".pdf"))
            if pdfs:
                return "scanned_claims/" + pdfs[0]
        except OSError:
            pass
    return "scanned_claims/doc_claim.pdf"


def test_live_http_path_accepts_untested_action_payloads():
    client = TestClient(app)

    response = client.post("/fraud_hunter/seed_range", json={"seed_min": None, "seed_max": None})
    response.raise_for_status()

    reset_payload = client.post("/reset")
    assert reset_payload.status_code == 200
    pdf_path = _first_pdf_in_case(reset_payload.json() or {})

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
            "think_trace": "<think>Validate HTTP serialization for code_act with multiline Python.</think>",
        },
        {
            "kind": "ocr_document",
            "pdf_path": pdf_path,
            "think_trace": "<think>Validate HTTP serialization for ocr_document.</think>",
        },
        {
            "kind": "compare_doc_vs_claim",
            "claim_id": "C003",
            "extracted_fields": {"claim_id": "C003", "amount": 480.0, "hcpcs_code": "99214"},
            "think_trace": "<think>Validate HTTP serialization for compare_doc_vs_claim.</think>",
        },
    ]

    for action in actions:
        response = client.post("/step", json={"action": action})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload.keys()) == {"observation", "reward", "done"}
