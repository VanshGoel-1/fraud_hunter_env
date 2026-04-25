"""
FastAPI application for the Fraud Hunter Environment.

Routes:
  /ws               — OpenEnv WebSocket protocol (authoritative on HF Spaces)
  /reset /step /state — OpenEnv HTTP convenience endpoints (local dev)
  /docs /redoc      — Swagger / ReDoc (from openenv-core)
  /dashboard        — Custom HTML monitoring UI (web/index.html)
  /metrics          — Server-Sent Events stream of episode metrics
  /leaderboard      — Top-10 episode rewards
  /health           — Health check (also registered by openenv-core at /health)

The FastAPI instance is built by `openenv.core.env_server.create_app` so the
WebSocket protocol matches what HF Spaces and training clients expect. Custom
routes are mounted afterward on the same app.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from fraud_hunter_env.models import FraudHunterAction, FraudHunterObservation
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment


# ── OpenEnv-compliant FastAPI app (provides /ws, /reset, /step, /state) ──────

app = create_app(
    env=lambda: FraudHunterEnvironment(),
    action_cls=FraudHunterAction,
    observation_cls=FraudHunterObservation,
    env_name="fraud_hunter_env",
)


# ── Episode metrics store (in-memory, shared across all sessions) ────────────

_episode_log: deque[dict[str, Any]] = deque(maxlen=500)
_sse_queues: list[asyncio.Queue] = []


def record_episode_metrics(metrics: dict[str, Any]) -> None:
    """Hook the environment can call after each terminal step to broadcast."""
    metrics["timestamp"] = time.time()
    _episode_log.append(metrics)
    for q in list(_sse_queues):
        try:
            q.put_nowait(metrics)
        except asyncio.QueueFull:
            pass


# ── Custom web UI: mount web/index.html ──────────────────────────────────────

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if _WEB_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_WEB_DIR), html=True),
        name="dashboard",
    )

    @app.get("/ui", include_in_schema=False)
    async def ui_redirect():
        """Short alias for the custom dashboard."""
        index = _WEB_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<h1>Dashboard not found.</h1>", status_code=404)
else:
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_missing():
        return HTMLResponse(
            "<h1>Dashboard not found. web/index.html is missing.</h1>",
            status_code=404,
        )


# ── Server-Sent Events: live metrics ─────────────────────────────────────────

@app.get("/metrics")
async def metrics_sse():
    """SSE endpoint: streams episode metrics as they arrive."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_queues.append(queue)

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _sse_queues:
                _sse_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Leaderboard ──────────────────────────────────────────────────────────────

@app.get("/leaderboard")
async def leaderboard():
    """Top 10 episodes by total reward."""
    sorted_eps = sorted(
        _episode_log, key=lambda e: e.get("episode_reward", 0), reverse=True
    )
    return sorted_eps[:10]


# ── Dashboard-specific health (openenv-core already registers /health) ──────

@app.get("/fraud_hunter/health")
async def fraud_hunter_health():
    return {
        "status": "healthy",
        "episodes_logged": len(_episode_log),
        "sse_clients": len(_sse_queues),
    }


def main():
    import uvicorn

    uvicorn.run(
        "fraud_hunter_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
