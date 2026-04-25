"""
Inference Smoke Test — Connects to the environment using FraudHunterEnv client 
and runs a sample agent loop.
"""

from client import FraudHunterEnv
from models import FraudHunterAction

def run_smoke_test():
    print("Starting Fraud Hunter Env Smoke Test...")
    
    # We can connect to an existing server or just use the Environment directly.
    # For a smoke test without starting the HTTP server, we can mock the client
    # or use the environment implementation. We'll use the raw environment 
    # to demonstrate inference.
    from server.fraud_hunter_env_environment import FraudHunterEnvironment
    env = FraudHunterEnvironment()
    
    obs = env.reset()
    print("\\n[EPISODE STARTED]")
    print(f"Briefing: {obs.case_brief}\\n")
    
    # Mock LLM actions based on the briefing. 
    # A real inference script would use an LLM router (e.g. vLLM MRV2) here.
    actions = [
        {"kind": "query_corporate", "entity_name": "Acme Shell LLC"},
        {"kind": "query_medicare", "beneficiary_id": "B001"},
        {"kind": "extract_entity", "extracted_name": "Fake Clinic Corp", "extracted_kind": "corporation"},
        {"kind": "submit_case", "case_summary": "Found Fake Clinic Corp.", "confidence": 0.85}
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
