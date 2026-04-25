"""
Episode-metrics broadcasting bus.

Decouples metric production (the environment's `on_episode_end` callback) from
metric consumption (the `/leaderboard` route + `/metrics` SSE stream + future
Redis-backed aggregator).

Current implementation is in-process and per-worker — the same correctness
caveat as `DifficultyManager`. When Phase 9.5 (Redis) lands, replace
`InMemoryMetricsBus` with a `RedisMetricsBus` behind the same interface.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Iterable, Protocol


class MetricsBus(Protocol):
    """Minimal interface — every bus must support these three operations."""

    def record(self, metrics: dict[str, Any]) -> None: ...
    def recent(self, limit: int) -> list[dict[str, Any]]: ...
    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]": ...
    def unsubscribe(self, queue: "asyncio.Queue[dict[str, Any]]") -> None: ...


class InMemoryMetricsBus:
    """Per-process bus. Bounded log + fan-out to active SSE subscribers."""

    def __init__(self, max_history: int = 500, queue_size: int = 50) -> None:
        self._log: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._subscribers: list[asyncio.Queue] = []
        self._queue_size = queue_size

    def record(self, metrics: dict[str, Any]) -> None:
        metrics = {**metrics, "timestamp": time.time()}
        self._log.append(metrics)
        for q in list(self._subscribers):
            try:
                q.put_nowait(metrics)
            except asyncio.QueueFull:
                # Slow consumer: drop the frame, never block the producer.
                pass

    def recent(self, limit: int) -> list[dict[str, Any]]:
        # Defensive copy; callers may sort/slice without mutating internal state.
        return list(self._log)[-limit:]

    def top_by(self, key: str, limit: int) -> list[dict[str, Any]]:
        return sorted(self._log, key=lambda m: m.get(key, 0), reverse=True)[:limit]

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def episode_count(self) -> int:
        return len(self._log)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)


__all__ = ["MetricsBus", "InMemoryMetricsBus"]
