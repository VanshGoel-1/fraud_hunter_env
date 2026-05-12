from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_hunter_env.server.app import app
import fraud_hunter_env.server.app as server_app


@pytest.fixture(autouse=True)
def _enable_online_rl():
    """Enable online RL for this test module (default is now False per N-09)."""
    original = server_app._ONLINE_RL_ENABLED
    server_app._ONLINE_RL_ENABLED = True
    yield
    server_app._ONLINE_RL_ENABLED = original


def test_online_rl_select_and_update_roundtrip():
    client = TestClient(app)

    reset_resp = client.post("/reset", json={})
    assert reset_resp.status_code == 200, reset_resp.text
    observation = (reset_resp.json() or {}).get("observation") or {}

    select_resp = client.post(
        "/fraud_hunter/agent_action_online",
        json={
            "observation": observation,
            "objective": "Investigate quickly and improve reward",
            "llm": {
                "enabled": False,
                "base_url": "",
                "model": "",
            },
        },
    )
    assert select_resp.status_code == 200, select_resp.text
    selected = select_resp.json() or {}

    token = selected.get("decision_token")
    action = selected.get("action")
    assert isinstance(token, str) and token
    assert isinstance(action, dict) and action.get("kind")

    step_resp = client.post("/step", json={"action": action})
    assert step_resp.status_code == 200, step_resp.text
    reward = float((step_resp.json() or {}).get("reward") or 0.0)

    update_resp = client.post(
        "/fraud_hunter/online_rl/update",
        json={"decision_token": token, "reward": reward},
    )
    assert update_resp.status_code == 200, update_resp.text
    update_payload = update_resp.json() or {}

    assert update_payload.get("status") == "updated"
    assert (update_payload.get("online_rl") or {}).get("updates", 0) >= 1
