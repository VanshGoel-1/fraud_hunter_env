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
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from openenv.core.env_server import create_app

from fraud_hunter_env.models import FraudHunterAction, FraudHunterObservation
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment


# ── Episode metrics store (in-memory, shared across all sessions) ────────────
# Defined before `create_app` so the env factory can close over the callback.

_episode_log: deque[dict[str, Any]] = deque(maxlen=500)
_sse_queues: list[asyncio.Queue] = []


def record_episode_metrics(metrics: dict[str, Any]) -> None:
    """Hook the environment calls after each terminal step to broadcast."""
    metrics["timestamp"] = time.time()
    _episode_log.append(metrics)
    for q in list(_sse_queues):
        try:
            q.put_nowait(metrics)
        except asyncio.QueueFull:
            pass


# ── OpenEnv-compliant FastAPI app (provides /ws, /reset, /step, /state) ──────

app = create_app(
    env=lambda: FraudHunterEnvironment(on_episode_end=record_episode_metrics),
    action_cls=FraudHunterAction,
    observation_cls=FraudHunterObservation,
    env_name="fraud_hunter_env",
)


# ── CORS ─────────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS is a comma-separated list. Default `*` keeps local dev
# frictionless; production deployments must set this explicitly.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ── API-key gate ─────────────────────────────────────────────────────────────
# FRAUD_HUNTER_API_KEYS is a comma-separated list of valid keys. When unset the
# gate is permissive (dev mode). When set, every non-public path requires
# X-API-Key.

_PUBLIC_PATH_PREFIXES = (
    "/health", "/docs", "/redoc", "/openapi.json",
    "/dashboard", "/ui", "/metrics", "/fraud_hunter/health",
)


def _load_api_keys() -> set[str]:
    raw = os.environ.get("FRAUD_HUNTER_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


# Cached at module import; restart the server to rotate keys.
_API_KEYS: set[str] = _load_api_keys()


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid X-API-Key when keys are configured."""

    async def dispatch(self, request: Request, call_next):
        if not _API_KEYS:
            return await call_next(request)
        # CORS preflights never carry custom headers; let them through so the
        # CORS middleware can answer with the appropriate Allow-* headers.
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PATH_PREFIXES):
            return await call_next(request)
        # WebSocket auth happens at handshake — Starlette routes ws via scope.
        if request.scope.get("type") == "websocket":
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if provided not in _API_KEYS:
            return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


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
