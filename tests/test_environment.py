import os
import pytest
from pathlib import Path
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
from fraud_hunter_env.models import FraudHunterAction, ActionKind

def test_environment_dynamic_generation():
    # 1. Initialize environment
    env = FraudHunterEnvironment()
    
    # 2. Reset generates a case dynamically
    obs = env.reset()
    assert obs.difficulty_tier == 1
    assert "case_id" in obs.info
    
    # 3. Verify directory structure
    case = env._case
    assert case is not None
    
    case_dir = case.db_path.parent
    assert case_dir.exists()
    assert case.db_path.exists()
    
    comms_dir = case_dir / "intercepted_comms"
    scanned_dir = case_dir / "scanned_claims"
    
    assert comms_dir.exists()
    assert scanned_dir.exists()
    
    # 4. Verify code sandbox can access the files
    # The agent should be able to list files in the case_dir and read a PDF
    test_code = """
files = listdir("scanned_claims")
if files:
    pdf_path = path_join("scanned_claims", files[0])
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""
        print("PDF Extracted:", len(text), "chars")
else:
    print("No PDFs found")
"""
    
    action = FraudHunterAction(
        think_trace="<think>testing sandbox</think>",
        kind=ActionKind.CODE_ACT,
        python_code=test_code
    )
    
    step_obs = env.step(action)
    assert "PDF Extracted:" in step_obs.tool_output or "No PDFs found" in step_obs.tool_output
    assert "SECURITY_VIOLATION" not in step_obs.tool_output
