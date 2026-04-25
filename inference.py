"""
Inference Smoke Test — drives the FraudHunterEnvironment **in-process** with
a minimal scripted action sequence. Does NOT spin up the HTTP server.

This is the single permitted in-process bypass of the OpenEnv contract: it's a
dev-loop sanity check, not the canonical client. For the real client-server
demo (and for any policy that should be reproducible by judges), use
``eval.py`` which drives the server through ``client.FraudHunterEnv`` over
HTTP/WS.

The scripted actions discover real entity / beneficiary identifiers from the
active case database (rather than hard-coding "Acme Shell LLC" / "B001" which
would never match a freshly-generated or bank-sampled case).
"""

from fraud_hunter_env.models import FraudHunterAction
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment


def _discover_seed_targets(env: FraudHunterEnvironment) -> tuple[str | None, str | None]:
    """Pull one entity name and one beneficiary id from the active case DB."""
    if env.case is None:
        return None, None
    conn = env.case.conn
    entity_name = None
    bene_id = None
    try:
        row = conn.execute(
            "SELECT entity_name FROM corporate_registry LIMIT 1"
        ).fetchone()
        if row:
            entity_name = row[0]
    except Exception:
        pass
    try:
        row = conn.execute(
            "SELECT DESYNPUF_ID FROM beneficiary_summary LIMIT 1"
        ).fetchone()
        if row:
            bene_id = row[0]
    except Exception:
        pass
    return entity_name, bene_id


def run_smoke_test():
    print("Starting Fraud Hunter Env Smoke Test...")
    env = FraudHunterEnvironment()
    obs = env.reset()
    print("\n[EPISODE STARTED]")
    print(f"Briefing: {obs.case_brief}\n")

    entity_name, bene_id = _discover_seed_targets(env)
    actions: list[dict] = []
    if entity_name:
        actions.append({"kind": "query_corporate", "entity_name": entity_name})
    if bene_id:
        actions.append({"kind": "query_medicare", "beneficiary_id": bene_id})
    actions.extend([
        {"kind": "extract_entity",
         "extracted_name": entity_name or "Unknown Corp",
         "extracted_kind": "corporation"},
        {"kind": "submit_case",
         "case_summary": f"Investigated {entity_name or 'unknown entity'}.",
         "confidence": 0.85},
    ])

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

    print("\n[EPISODE FINISHED]")


if __name__ == "__main__":
    run_smoke_test()
