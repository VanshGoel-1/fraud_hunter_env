"""Standalone HTTP action-surface validator (no pytest required).

Checks that the previously unverified actions serialize and pass through
`/step` without 422 schema errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `fraud_hunter_env.*` imports work when this script is run directly
# from the repository root without requiring manual PYTHONPATH exports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from fastapi.testclient import TestClient

from fraud_hunter_env.server.app import app


def _post_remote(base_url: str, path: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{base_url}{path}", json=payload or {}, timeout=10)
    response.raise_for_status()
    return response.json()


def _step_remote(base_url: str, action: dict) -> dict:
    response = requests.post(f"{base_url}/step", json={"action": action}, timeout=10)
    if response.status_code == 422:
        raise RuntimeError(f"422 from /step for action {action['kind']}: {response.text}")
    response.raise_for_status()
    payload = response.json()
    expected = {"observation", "reward", "done"}
    if set(payload.keys()) != expected:
        raise RuntimeError(
            f"Unexpected /step payload keys for action {action['kind']}: {sorted(payload.keys())}"
        )
    return payload


def _post_local(client: TestClient, path: str, payload: dict | None = None) -> dict:
    response = client.post(path, json=payload or {})
    response.raise_for_status()
    return response.json()


def _step_local(client: TestClient, action: dict) -> dict:
    response = client.post("/step", json={"action": action})
    if response.status_code == 422:
        raise RuntimeError(f"422 from /step for action {action['kind']}: {response.text}")
    response.raise_for_status()
    payload = response.json()
    expected = {"observation", "reward", "done"}
    if set(payload.keys()) != expected:
        raise RuntimeError(
            f"Unexpected /step payload keys for action {action['kind']}: {sorted(payload.keys())}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--remote", action="store_true", help="Use a running server instead of in-process FastAPI client")
    parser.add_argument("--seed-min", type=int, default=8001)
    parser.add_argument("--seed-max", type=int, default=10000)
    args = parser.parse_args()

    if args.remote:
        post = lambda path, payload=None: _post_remote(args.base_url, path, payload)
        step = lambda action: _step_remote(args.base_url, action)
    else:
        client = TestClient(app)
        post = lambda path, payload=None: _post_local(client, path, payload)
        step = lambda action: _step_local(client, action)

    post(
        "/fraud_hunter/seed_range",
        {"seed_min": args.seed_min, "seed_max": args.seed_max},
    )
    post("/reset")

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
            "pdf_path": "scanned_claims/doc_claim.pdf",
            "think_trace": "<think>Validate HTTP serialization for ocr_document.</think>",
        },
        {
            "kind": "compare_doc_vs_claim",
            "claim_id": "C003",
            "extracted_fields": {"claim_id": "C003", "amount": 480.0, "hcpcs_code": "99214"},
            "think_trace": "<think>Validate HTTP serialization for compare_doc_vs_claim.</think>",
        },
    ]

    results: dict[str, str] = {}
    for action in actions:
        step(action)
        results[action["kind"]] = "ok"

    print(json.dumps({"status": "ok", "validated": results}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
