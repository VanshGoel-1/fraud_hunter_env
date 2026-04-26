"""Inference smoke test over the network OpenEnv client contract.

This script intentionally avoids in-process environment imports and drives the
server exclusively through FraudHunterEnv (HTTP/WS).
"""

from __future__ import annotations

import argparse
import urllib.request

from fraud_hunter_env.client import FraudHunterEnv
from fraud_hunter_env.models import FraudHunterAction


def _ensure_server_up(base_url: str) -> None:
    request = urllib.request.Request(f"{base_url}/health", method="GET")
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"/health returned {response.status}")


def run_smoke_test(base_url: str):
    print("Starting Fraud Hunter Env Smoke Test over client/server API...")

    with FraudHunterEnv(base_url=base_url).sync() as env:
        obs = env.reset().observation
        print("\n[EPISODE STARTED]")
        print(f"Briefing: {obs.case_brief}\n")

        actions: list[dict] = [
            {
                "kind": "code_act",
                "python_code": "files = listdir('intercepted_comms')\nprint(files[:3])",
                "think_trace": "<think>Enumerate intercepted communications evidence.</think>",
            },
            {
                "kind": "query_medicare",
                "claim_id": "C003",
                "think_trace": "<think>Probe known claim identifier path for contract sanity.</think>",
            },
            {
                "kind": "submit_case",
                "case_summary": "Inference smoke validation run.",
                "confidence": 0.5,
                "think_trace": "<think>Terminate smoke run after validating tool surface.</think>",
            },
        ]

        for act_payload in actions:
            print(f"-> Agent Action: {act_payload}")
            try:
                action = FraudHunterAction.model_validate(act_payload)
                step = env.step(action)
                obs = step.observation
                print(f"<- Env Response (Reward: {float(step.reward or 0.0):.2f}, Done: {step.done})")
                if obs.tool_output:
                    print(f"   Tool Output: {obs.tool_output[:160]}...")
                if obs.grader_feedback:
                    print(f"   Feedback: {obs.grader_feedback}")
                if step.done:
                    break
            except Exception as e:
                print(f"Action validation failed: {e}")
                break

        print("\n[EPISODE FINISHED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    _ensure_server_up(args.base_url)
    run_smoke_test(args.base_url)
