"""
Inference Smoke Test — Connects to the environment using FraudHunterEnv client 
and runs a sample agent loop.

Run from the repo's parent directory:
    python -m fraud_hunter_env.inference
"""

from fraud_hunter_env.client import FraudHunterEnv  # noqa: F401  (kept for parity with HTTP path)
from fraud_hunter_env.models import FraudHunterAction

def run_smoke_test():
    print("Starting Fraud Hunter Env Smoke Test...")

    # We can connect to an existing server or just use the Environment directly.
    # For a smoke test without starting the HTTP server, we can mock the client
    # or use the environment implementation. We'll use the raw environment
    # to demonstrate inference.
    from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
    env = FraudHunterEnvironment()
    
    obs = env.reset()
    print("\\n[EPISODE STARTED]")
    print(f"Briefing: {obs.case_brief}\\n")

    # Pull real entity / beneficiary IDs from this episode's DB so the smoke
    # test exercises the happy path even though tier-1 cases are randomised.
    case_conn = env._case.conn  # noqa: SLF001 — smoke test internal access
    corp_row = case_conn.execute(
        "SELECT entity_name FROM corporate_registry LIMIT 1"
    ).fetchone()
    bene_row = case_conn.execute(
        "SELECT DESYNPUF_ID FROM beneficiary_summary LIMIT 1"
    ).fetchone()
    sample_entity = corp_row[0] if corp_row else "Acme Shell LLC"
    sample_bene = bene_row[0] if bene_row else "BENE_001"

    actions = [
        {"kind": "query_corporate", "entity_name": sample_entity},
        {"kind": "query_medicare", "beneficiary_id": sample_bene},
        {"kind": "extract_entity", "extracted_name": sample_entity, "extracted_kind": "corporation"},
        {"kind": "submit_case", "case_summary": f"Flagged {sample_entity} during smoke test.", "confidence": 0.85},
    ]
    
    for act_payload in actions:
        print(f"-> Agent Action: {act_payload}")
        try:
            action = FraudHunterAction.model_validate(act_payload)
            obs = env.step(action)
            
            print(f"<- Env Response (Reward: {obs.reward:.2f}, Done: {obs.done})")
            if obs.tool_output:
                print(f"   Tool Output: {obs.tool_output[:100]}...")
            if obs.grader_feedback:
                print(f"   Feedback: {obs.grader_feedback}")
                
            if obs.done:
                break
        except Exception as e:
            print(f"Action validation failed: {e}")
            break

    print("\\n[EPISODE FINISHED]")

if __name__ == "__main__":
    run_smoke_test()
