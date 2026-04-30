"""Unit tests for fraud_hunter_env.replay.verify_replay_chain."""

import pytest

from fraud_hunter_env.replay import verify_replay_chain


def test_empty_chain_is_ok():
    ok, reason = verify_replay_chain([])
    assert ok is True
    assert reason == "ok"


def test_single_step_is_ok():
    ok, reason = verify_replay_chain([{"replay_hash": "abc", "replay_prev_hash": None}])
    assert ok is True
    assert reason == "ok"


def test_well_formed_chain_passes():
    history = [
        {"replay_hash": "h1", "replay_prev_hash": None},
        {"replay_hash": "h2", "replay_prev_hash": "h1"},
        {"replay_hash": "h3", "replay_prev_hash": "h2"},
    ]
    ok, reason = verify_replay_chain(history)
    assert ok is True
    assert reason == "ok"


def test_broken_chain_detected_with_step_in_reason():
    history = [
        {"replay_hash": "h1", "replay_prev_hash": None},
        {"replay_hash": "h2", "replay_prev_hash": "h1"},
        {"replay_hash": "h3", "replay_prev_hash": "WRONG"},
    ]
    ok, reason = verify_replay_chain(history)
    assert ok is False
    assert "step 2" in reason
    assert "WRONG" in reason


def test_missing_prev_hash_is_break():
    history = [
        {"replay_hash": "h1", "replay_prev_hash": None},
        {"replay_hash": "h2"},  # missing replay_prev_hash entirely
    ]
    ok, reason = verify_replay_chain(history)
    assert ok is False
    assert "step 1" in reason


def test_live_env_chain_verifies():
    """Drive a 3-step episode through the in-process env and verify the chain
    is internally consistent end-to-end."""
    from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
    from fraud_hunter_env.models import FraudHunterAction, ActionKind

    env = FraudHunterEnvironment()
    try:
        infos = [env.reset().info]
        for _ in range(2):
            obs = env.step(FraudHunterAction(
                kind=ActionKind.QUERY_CORPORATE,
                entity_name="Acme",
                think_trace="<think>probe</think>",
            ))
            infos.append(obs.info)
        ok, reason = verify_replay_chain(infos)
        assert ok, f"chain failed: {reason}"
        assert reason == "ok"
    finally:
        env.close()
