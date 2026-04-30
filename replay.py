"""Replay hash chain verification.

The environment in ``server/fraud_hunter_env_environment.py`` emits a SHA-256
hash chain inside each observation's ``info`` dict (``replay_hash`` +
``replay_prev_hash``). This module re-walks a captured chain and asserts that
``info[i]["replay_prev_hash"] == info[i-1]["replay_hash"]`` for every i,
giving evaluators a way to detect replay tampering or out-of-order info
records.

Pure function — no I/O, no env state. Safe to call from any context (eval
harness, tests, ad-hoc scripts).
"""

from __future__ import annotations

from typing import Sequence


def verify_replay_chain(info_history: Sequence[dict]) -> tuple[bool, str]:
    """Walk the chain and confirm each entry's prev_hash matches its predecessor.

    Returns ``(True, "ok")`` for empty or single-step histories (nothing to
    chain against). Returns ``(False, reason)`` at the first mismatch with a
    human-readable description of which step broke and the conflicting hashes.

    The check is intentionally schema-light: any ``info`` dict with the two
    keys works, and ``None`` for ``replay_prev_hash`` on the first step is
    accepted (the env emits ``None`` on reset).
    """
    if len(info_history) <= 1:
        return True, "ok"
    for i in range(1, len(info_history)):
        prev_hash = info_history[i - 1].get("replay_hash")
        current_prev = info_history[i].get("replay_prev_hash")
        if prev_hash != current_prev:
            return False, (
                f"chain break at step {i}: "
                f"prev.replay_hash={prev_hash!r} "
                f"!= current.replay_prev_hash={current_prev!r}"
            )
    return True, "ok"


__all__ = ["verify_replay_chain"]
