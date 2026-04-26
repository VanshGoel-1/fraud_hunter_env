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


# ── Episode metrics store (in-memory + JSONL persistence) ────────────────────
# Persistence lets the dashboard show real history across server restarts
# and on a fresh HF Spaces container (if assets/metrics_history.jsonl is
# shipped in the image). Set FRAUD_HUNTER_METRICS_HISTORY="" to disable,
# or to a custom path to relocate. Multi-worker correctness still requires
# Redis (Phase 9.5).

import os as _os

_METRICS_PERSIST_DEFAULT = (
    Path(__file__).resolve().parent.parent / "assets" / "metrics_history.jsonl"
)
_metrics_persist_env = _os.environ.get("FRAUD_HUNTER_METRICS_HISTORY")
_metrics_persist_path = (
    None
    if _metrics_persist_env == ""
    else Path(_metrics_persist_env) if _metrics_persist_env
    else _METRICS_PERSIST_DEFAULT
)

metrics_bus = InMemoryMetricsBus(persist_path=_metrics_persist_path)


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
# When installed into site-packages, __file__ is under the wheel and `web/`
# (a top-level repo folder) is NOT alongside server/. Try several candidate
# roots so the dashboard works in: editable install, wheel install, Docker
# (COPY . /app/env), and the openenv-base build.
def _find_dir(name: str) -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / name,                    # editable / flat layout
        here.parent.parent.parent / name,             # one level up (src/)
        Path("/app/env") / name,                      # Dockerfile layout
        Path.cwd() / name,                            # CWD fallback
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

_WEB_DIR = _find_dir("web")
_ASSETS_DIR = _find_dir("assets")

if _ASSETS_DIR is not None:
    app.mount(
        "/assets",
        StaticFiles(directory=str(_ASSETS_DIR)),
        name="assets",
    )

if _WEB_DIR is not None:
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


# ── Metrics history (bootstrap source for the dashboard) ─────────────────────

@app.get("/metrics/history")
async def metrics_history(limit: int = 100):
    """Return the most recent N episode metrics dicts. The dashboard fetches
    this on initial load so the KPIs/charts populate from the bus history
    even when no live client is currently driving the env."""
    return metrics_bus.recent(limit)


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
