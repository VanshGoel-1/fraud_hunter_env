import os
import pytest
from pathlib import Path
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
from fraud_hunter_env.models import FraudHunterAction, ActionKind, EntityKind

def test_environment_dynamic_generation(tmp_path):
    # 1. Initialize environment with a deliberately non-existent bank dir so
    #    reset() falls through to on-the-fly generation (not bank-pick). Stale
    #    cases in the real bank may pre-date the multimodal compiler and lack
    #    intercepted_comms/, which would spuriously fail the structure asserts
    #    below.
    env = FraudHunterEnvironment(case_bank_dir=str(tmp_path / "no_bank_here"))
    
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

def test_evidence_graph_entities_are_typed_dicts():
    """Confirmed entity extractions should appear as {name, kind} dicts."""
    env = FraudHunterEnvironment()
    try:
        env.reset()
        # Probe corporate_registry for a real entity name from the active case.
        probe = FraudHunterAction(
            kind=ActionKind.SQL_QUERY,
            sql_statement="SELECT entity_name FROM corporate_registry LIMIT 1",
            think_trace="<think>probe registry</think>",
        )
        probe_obs = env.step(probe)
        # Pull the first row of tool_output (header line is skipped); fall back
        # to the demo-case ground-truth name when parsing the table fails.
        name = "Acme Shell LLC"
        if probe_obs.tool_output:
            lines = [ln.strip() for ln in probe_obs.tool_output.splitlines() if ln.strip()]
            if len(lines) >= 2:
                name = lines[1]

        ext = FraudHunterAction(
            kind=ActionKind.EXTRACT_ENTITY,
            extracted_name=name,
            extracted_kind=EntityKind.CORPORATION,
            think_trace="<think>flag this corporation</think>",
        )
        obs = env.step(ext)

        assert obs.evidence_graph is not None
        entities = obs.evidence_graph["entities"]
        assert isinstance(entities, list)
        # Each entry (if any) must be a dict with both `name` and `kind`.
        for item in entities:
            assert isinstance(item, dict)
            assert "name" in item and "kind" in item
            assert isinstance(item["name"], str) and isinstance(item["kind"], str)
    finally:
        env.close()


def test_agentic_recall_ignores_sql_substrings_without_access():
    env = FraudHunterEnvironment(case_bank_dir=None)
    env.reset()
    obs = env.step(FraudHunterAction.model_validate({
        "kind": "sql_query",
        "sql_statement": "SELECT 'corporate_registry', 'beneficiary_summary'",
        "think_trace": "<think>Test literal strings without touching any tables.</think>",
    }))
    assert obs.info is not None
    assert obs.info["agentic_recall"] == 0.0
