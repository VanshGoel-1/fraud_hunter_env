"""Shared HTTP helpers for smoke scripts and tests."""

from __future__ import annotations

import urllib.request
from typing import Any

EXPECTED_STEP_KEYS = {"observation", "reward", "done"}


def ensure_server_up(base_url: str, timeout: float = 3) -> None:
    request = urllib.request.Request(f"{base_url}/health", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"/health returned {response.status}")


def assert_step_payload(payload: dict[str, Any], action_kind: str | None = None) -> None:
    if set(payload.keys()) != EXPECTED_STEP_KEYS:
        context = f" for action {action_kind}" if action_kind else ""
        raise RuntimeError(
            f"Unexpected /step payload keys{context}: {sorted(payload.keys())}"
        )


def post_testclient_json(client: Any, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post(path, json=payload or {})
    response.raise_for_status()
    return response.json()


def step_testclient_json(client: Any, action: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/step", json={"action": action})
    if response.status_code == 422:
        raise RuntimeError(f"422 from /step for action {action['kind']}: {response.text}")
    response.raise_for_status()
    payload = response.json()
    assert_step_payload(payload, str(action.get("kind", "")))
    return payload


def post_requests_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    import requests

    response = requests.post(f"{base_url}{path}", json=payload or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def step_requests_json(
    base_url: str,
    action: dict[str, Any],
    timeout: float = 10,
) -> dict[str, Any]:
    import requests

    response = requests.post(f"{base_url}/step", json={"action": action}, timeout=timeout)
    if response.status_code == 422:
        raise RuntimeError(f"422 from /step for action {action['kind']}: {response.text}")
    response.raise_for_status()
    payload = response.json()
    assert_step_payload(payload, str(action.get("kind", "")))
    return payload
