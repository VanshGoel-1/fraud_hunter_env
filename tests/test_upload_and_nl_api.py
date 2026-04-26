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
