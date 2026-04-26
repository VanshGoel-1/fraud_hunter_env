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
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol


class MetricsBus(Protocol):
    """Minimal interface — every bus must support these three operations."""

    def record(self, metrics: dict[str, Any]) -> None: ...
    def recent(self, limit: int) -> list[dict[str, Any]]: ...
    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]": ...
    def unsubscribe(self, queue: "asyncio.Queue[dict[str, Any]]") -> None: ...


class InMemoryMetricsBus:
    """Per-process bus. Bounded log + fan-out to active SSE subscribers.

    Optional ``persist_path`` enables append-only JSONL persistence so the
    dashboard sees historical episodes after a server restart (and on a
    fresh HF Spaces container, if the JSONL is shipped in the image).
    """

    def __init__(
        self,
        max_history: int = 500,
        queue_size: int = 50,
        persist_path: Optional[str | Path] = None,
    ) -> None:
        self._log: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._subscribers: list[asyncio.Queue] = []
        self._queue_size = queue_size
        self._persist_path: Optional[Path] = (
            Path(persist_path) if persist_path else None
        )
        self._persist_lock = threading.Lock()
        if self._persist_path is not None:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Replay JSONL into the in-memory log on startup. Best-effort —
        a corrupt line skips silently rather than blocking server boot."""
        if self._persist_path is None or not self._persist_path.is_file():
            return
        try:
            for line in self._persist_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._log.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass

    def _persist_one(self, metrics: dict[str, Any]) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_lock, self._persist_path.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
        except OSError:
            # Persistence is best-effort; never block episode flow on disk IO.
            pass

    def record(self, metrics: dict[str, Any]) -> None:
        metrics = {**metrics, "timestamp": time.time()}
        self._log.append(metrics)
        self._persist_one(metrics)
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
