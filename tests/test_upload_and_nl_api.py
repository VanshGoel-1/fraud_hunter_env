from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from fraud_hunter_env.server import app as server_app


def test_upload_csv_dataset_roundtrip(tmp_path):
    server_app._UPLOAD_ENABLED = True
    server_app._UPLOAD_DIR = tmp_path / "uploads"
    server_app._UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    client = TestClient(server_app.app)

    payload = "id,value\n1,foo\n2,bar\n".encode("utf-8")
    response = client.post(
        "/fraud_hunter/upload_dataset",
        data={"dataset_name": "claims_batch_1", "extract_zip": "false"},
        files={"file": ("claims.csv", payload, "text/csv")},
    )

    assert response.status_code == 200, response.text
    body = response.json() or {}
    assert body.get("status") == "uploaded"
    assert body.get("dataset") == "claims_batch_1"
    assert body.get("extracted_count") == 0

    stored = server_app._UPLOAD_DIR / body["stored_file"]
    assert stored.exists()
    assert stored.read_text(encoding="utf-8") == payload.decode("utf-8")


def test_upload_multiple_csv_files_roundtrip(tmp_path):
    server_app._UPLOAD_ENABLED = True
    server_app._UPLOAD_DIR = tmp_path / "uploads"
    server_app._UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    client = TestClient(server_app.app)
    first = "id,value\n1,foo\n".encode("utf-8")
    second = "id,value\n2,bar\n".encode("utf-8")
    response = client.post(
        "/fraud_hunter/upload_dataset",
        data={"dataset_name": "claims_batch_multi", "extract_zip": "false"},
        files=[
            ("file", ("claims_a.csv", first, "text/csv")),
            ("file", ("claims_b.csv", second, "text/csv")),
        ],
    )

    assert response.status_code == 200, response.text
    body = response.json() or {}
    assert body.get("file_count") == 2
    stored_files = body.get("stored_files") or []
    assert len(stored_files) == 2
    assert (server_app._UPLOAD_DIR / stored_files[0]).read_text(encoding="utf-8") == first.decode("utf-8")
    assert (server_app._UPLOAD_DIR / stored_files[1]).read_text(encoding="utf-8") == second.decode("utf-8")


def test_upload_zip_blocks_path_traversal(tmp_path):
    server_app._UPLOAD_ENABLED = True
    server_app._UPLOAD_DIR = tmp_path / "uploads"
    server_app._UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    malicious_zip = io.BytesIO()
    with zipfile.ZipFile(malicious_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.csv", "id,value\n1,pwned\n")
    malicious_zip.seek(0)

    client = TestClient(server_app.app)
    response = client.post(
        "/fraud_hunter/upload_dataset",
        data={"dataset_name": "bad_zip", "extract_zip": "true"},
        files={"file": ("bad.zip", malicious_zip.getvalue(), "application/zip")},
    )

    assert response.status_code == 400, response.text
    assert "path traversal" in str((response.json() or {}).get("detail", "")).lower()


def test_nl_action_requires_user_message():
    client = TestClient(server_app.app)
    response = client.post(
        "/fraud_hunter/nl_action",
        json={
            "observation": {"case_brief": "brief", "step_count": 0},
            "user_message": "",
        },
    )

    assert response.status_code == 400, response.text
    assert (response.json() or {}).get("detail") == "user_message is required"


def test_nl_action_falls_back_when_agent_disabled():
    server_app._AGENT_MODEL = ""
    server_app._AGENT_BASE_URL = ""

    client = TestClient(server_app.app)
    response = client.post(
        "/fraud_hunter/nl_action",
        json={
            "observation": {
                "step_count": 0,
                "tool_output": "",
                "grader_feedback": "",
            },
            "objective": "Investigate medicare fraud efficiently.",
            "user_message": "Find the most suspicious next move.",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json() or {}
    assert isinstance(body.get("action"), dict)
    assert body.get("provider") == "heuristic-fallback"
    assert "llm_error" in body


def test_stateful_http_session_reset_then_step_returns_output():
    client = TestClient(server_app.app)
    headers = {"x-session-id": "test-session-a"}

    reset = client.post("/fraud_hunter/session_reset", headers=headers)
    assert reset.status_code == 200, reset.text

    step = client.post(
        "/fraud_hunter/session_step",
        headers=headers,
        json={
            "action": {
                "kind": "sql_query",
                "sql_statement": "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                "think_trace": "<think>Inspect schema first.</think>",
            }
        },
    )
    assert step.status_code == 200, step.text
    body = step.json() or {}
    obs = body.get("observation") or {}
    assert obs.get("step_count") == 1
    assert isinstance(obs.get("tool_output"), str)
    assert len((obs.get("tool_output") or "").strip()) > 0
