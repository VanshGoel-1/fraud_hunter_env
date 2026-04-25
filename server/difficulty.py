"""
RLVE Adaptive Difficulty Manager for the Fraud Hunter Environment.

Implements Reinforcement Learning with Adaptive Verifiable Environments:
the case bank is sampled by difficulty tier based on the agent's rolling
average reward, keeping the agent at its capability frontier.

Tier 1 (Easy):   Single dead-patient-claim or duplicate billing. 2 entities, 1 contradiction.
Tier 2 (Medium): Dead patient + duplicate, 1-layer shell, 4 entities.
Tier 3 (Hard):   AKS kickback structure, 2-layer shell, 6 entities, red herrings.
Tier 4 (Expert): PPP fraud + healthcare fraud, 3-layer shell, 8 entities, 3 typologies.
Tier 5 (Elite):  Full multi-sector fraud (healthcare+contracting+PPP), 4-layer shell,
                 10 entities, 5 typologies, planted red herrings, Benford anomaly.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class AgentProfile:
    """Rolling performance tracker for a single agent/session."""
    session_id: str
    reward_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    episode_count: int = 0
    current_tier: int = 1
    best_reward: float = float("-inf")
    total_format_errors: int = 0

    def record_episode(self, total_reward: float, format_errors: int = 0) -> None:
        self.reward_history.append(total_reward)
        self.episode_count += 1
        self.total_format_errors += format_errors
        if total_reward > self.best_reward:
            self.best_reward = total_reward
        self._update_tier()

    def _update_tier(self) -> None:
        if len(self.reward_history) < 3:
            return  # Need at least 3 episodes to assess
        avg = statistics.mean(self.reward_history)
        # ELO-inspired thresholds
        if avg >= 800.0:
            self.current_tier = 5
        elif avg >= 400.0:
            self.current_tier = 4
        elif avg >= 150.0:
            self.current_tier = 3
        elif avg >= 50.0:
            self.current_tier = 2
        else:
            self.current_tier = 1

    @property
    def rolling_avg(self) -> float:
        if not self.reward_history:
            return 0.0
        return statistics.mean(self.reward_history)


class DifficultyManager:
    """
    Global RLVE manager. Maps session IDs to AgentProfiles.
    The environment calls `get_tier(session_id)` when sampling a new case.
    """

    _TIER_DESCRIPTIONS = {
        1: "Beginner: single dead-patient-claim or duplicate bill, 2 entities",
        2: "Intermediate: dead patient + duplicate, 1-layer shell, 4 entities",
        3: "Advanced: AKS kickback, 2-layer shell, 6 entities, red herrings",
        4: "Expert: PPP + healthcare, 3-layer shell, 8 entities, 3 typologies",
        5: "Elite: multi-sector fraud, 4-layer shell, 10 entities, 5 typologies",
    }

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        # In-process correctness only. Multi-worker correctness requires Redis
        # (deferred to Phase 9.5).
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> AgentProfile:
        with self._lock:
            if session_id not in self._profiles:
                self._profiles[session_id] = AgentProfile(session_id=session_id)
            return self._profiles[session_id]

    def get_tier(self, session_id: str) -> int:
        return self.get_or_create(session_id).current_tier

    def record_episode(
        self,
        session_id: str,
        total_reward: float,
        format_errors: int = 0,
    ) -> int:
        """Record episode result and return the NEW tier for next episode."""
        profile = self.get_or_create(session_id)
        with self._lock:
            profile.record_episode(total_reward, format_errors)
            return profile.current_tier

    def get_stats(self, session_id: str) -> dict:
        profile = self.get_or_create(session_id)
        return {
            "session_id": session_id,
            "current_tier": profile.current_tier,
            "tier_description": self._TIER_DESCRIPTIONS[profile.current_tier],
            "rolling_avg_reward": round(profile.rolling_avg, 2),
            "episode_count": profile.episode_count,
            "best_reward": round(profile.best_reward, 2),
            "total_format_errors": profile.total_format_errors,
        }


# Global singleton — shared across all environment instances in one server process
_global_difficulty_manager = DifficultyManager()


def get_difficulty_manager() -> DifficultyManager:
    return _global_difficulty_manager
