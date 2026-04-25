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
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from openenv.core.env_server import create_app

from fraud_hunter_env import config
from fraud_hunter_env.models import FraudHunterAction, FraudHunterObservation
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
from fraud_hunter_env.server.metrics_bus import InMemoryMetricsBus


# ── Episode metrics store (in-memory, shared across all sessions) ────────────
# NOTE: This is per-worker. For multi-worker uvicorn, migrate to Redis (Phase 9.5).

metrics_bus = InMemoryMetricsBus()


def record_episode_metrics(metrics: dict[str, Any]) -> None:
    """Backwards-compatible alias — forwards to the new MetricsBus."""
    metrics_bus.record(metrics)


# ── OpenEnv-compliant FastAPI app (provides /ws, /reset, /step, /state) ──────
# Wire `on_episode_end` so terminal-step metrics flow into _episode_log + SSE.

app = create_app(
    env=lambda: FraudHunterEnvironment(on_episode_end=metrics_bus.record),
    action_cls=FraudHunterAction,
    observation_cls=FraudHunterObservation,
    env_name="fraud_hunter_env",
)


# ── CORS allowlist (env-driven) ──────────────────────────────────────────────
# Defaults are dev-friendly; tighten via ALLOWED_ORIGINS="https://a.com,https://b.com".

_DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
_allowed_origins = config.allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── API-key authentication (env-driven) ──────────────────────────────────────
# Cache keys at module-load time. To rotate, restart the server.
# Public prefixes (health, docs, dashboard, metrics SSE) bypass auth so the
# UI/operators can probe the service without a key.

_PUBLIC_PREFIXES = config.PUBLIC_ROUTE_PREFIXES


def _load_api_keys() -> set[str]:
    """Backwards-compatible alias — forwards to fraud_hunter_env.config."""
    return config.api_keys()


_API_KEYS: set[str] = config.api_keys()


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce X-API-Key on all non-public, non-WS HTTP routes."""

    async def dispatch(self, request: Request, call_next):
        # Never authenticate WebSocket frames or CORS preflight here.
        if request.scope.get("type") == "websocket":
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        # If no keys are configured, auth is disabled (dev mode).
        if not _API_KEYS:
            return await call_next(request)
        path = request.url.path
        if any(path == p or path.startswith(p + "/") or path == p for p in _PUBLIC_PREFIXES):
            return await call_next(request)
        provided = request.headers.get("x-api-key")
        if not provided or provided not in _API_KEYS:
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
    queue = metrics_bus.subscribe()

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            metrics_bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Leaderboard ──────────────────────────────────────────────────────────────

@app.get("/leaderboard")
async def leaderboard():
    """Top 10 episodes by total reward."""
    return metrics_bus.top_by("episode_reward", limit=10)


# ── Dashboard-specific health (openenv-core already registers /health) ──────

@app.get("/fraud_hunter/health")
async def fraud_hunter_health():
    return {
        "status": "healthy",
        "episodes_logged": metrics_bus.episode_count(),
        "sse_clients": metrics_bus.subscriber_count(),
    }


def main():
    import uvicorn

    uvicorn.run(
        "fraud_hunter_env.server.app:app",
        host=config.server_host(),
        port=config.server_port(),
        reload=False,
    )


if __name__ == "__main__":
    main()
