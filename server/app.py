"""
FastAPI application for the Fraud Hunter Environment.

Routes:
  /ws               — OpenEnv WebSocket protocol (authoritative on HF Spaces)
  /reset /step /state — OpenEnv HTTP convenience endpoints (local dev)
    /dashboard/config — Dashboard runtime configuration (routes/auth flags)
    /ui/config        — Stable dashboard configuration endpoint
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
import contextvars
import json
import math
import mimetypes
import random
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from fastapi import File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from openenv.core.env_server import create_app

from fraud_hunter_env import config
from fraud_hunter_env.models import FraudHunterAction, FraudHunterObservation
from fraud_hunter_env.server.fraud_hunter_env_environment import FraudHunterEnvironment
from fraud_hunter_env.server.metrics_bus import InMemoryMetricsBus
from fraud_hunter_env.server.online_rl import OnlineRLPolicy


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
_SEED_RANGE_BY_SCOPE: dict[str, tuple[int, int] | None] = {"global": None}
_SEED_SCOPE_LOCK = threading.Lock()
_HTTP_SESSION_ENVS: dict[str, FraudHunterEnvironment] = {}
_HTTP_SESSION_LAST_ACCESS: dict[str, float] = {}
_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_SESSION_MAX = 1000
_HTTP_SESSION_IDLE_SECS = 3600
_REQUEST_SEED_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_seed_scope",
    default="global",
)


def _request_seed_scope(request: Request) -> str:
    explicit = (request.headers.get("x-seed-scope") or "").strip()
    if explicit:
        return explicit
    api_key = (request.headers.get("x-api-key") or "").strip()
    if api_key:
        return f"api:{api_key}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _active_seed_range() -> tuple[int, int] | None:
    scope = _REQUEST_SEED_SCOPE.get()
    with _SEED_SCOPE_LOCK:
        scoped = _SEED_RANGE_BY_SCOPE.get(scope)
        if scoped is not None:
            return scoped
        return _SEED_RANGE_BY_SCOPE.get("global")


def _http_session_key(request: Request) -> str:
    explicit = (request.headers.get("x-session-id") or "").strip()
    if explicit:
        return explicit
    return _request_seed_scope(request)


def _evict_idle_http_sessions() -> None:
    cutoff = time.time() - _HTTP_SESSION_IDLE_SECS
    stale = [k for k, t in _HTTP_SESSION_LAST_ACCESS.items() if t < cutoff]
    for k in stale:
        env = _HTTP_SESSION_ENVS.pop(k, None)
        _HTTP_SESSION_LAST_ACCESS.pop(k, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    if len(_HTTP_SESSION_ENVS) > _HTTP_SESSION_MAX:
        oldest = sorted(_HTTP_SESSION_LAST_ACCESS, key=_HTTP_SESSION_LAST_ACCESS.get)
        for k in oldest[: len(_HTTP_SESSION_ENVS) - _HTTP_SESSION_MAX]:
            env = _HTTP_SESSION_ENVS.pop(k, None)
            _HTTP_SESSION_LAST_ACCESS.pop(k, None)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


def _get_or_create_http_env(request: Request) -> FraudHunterEnvironment:
    key = _http_session_key(request)
    with _HTTP_SESSION_LOCK:
        _HTTP_SESSION_LAST_ACCESS[key] = time.time()
        existing = _HTTP_SESSION_ENVS.get(key)
        if existing is not None:
            return existing
        _evict_idle_http_sessions()
        env = FraudHunterEnvironment(
            on_episode_end=metrics_bus.record,
            case_seed_range=_active_seed_range(),
        )
        _HTTP_SESSION_ENVS[key] = env
        return env


# ── OpenEnv-compliant FastAPI app (provides /ws, /reset, /step, /state) ──────
# Wire `on_episode_end` so terminal-step metrics flow into _episode_log + SSE.

app = create_app(
    env=lambda: FraudHunterEnvironment(
        on_episode_end=metrics_bus.record,
        case_seed_range=_active_seed_range(),
    ),
    action_cls=FraudHunterAction,
    observation_cls=FraudHunterObservation,
    env_name="fraud_hunter_env",
)

# ── Override stateless OpenEnv /reset and /step with session-aware versions ──
# The framework's built-in /reset and /step create and destroy a fresh
# environment on every call (stateless by OpenEnv design). That means /step
# always runs on an uninitialized env → reward=0, tool_output=None.
# We remove those routes and re-register them against _HTTP_SESSION_ENVS so
# multi-step HTTP sessions work correctly. WebSocket (/ws) is unaffected.
_router_routes = app.router.routes
for _i in range(len(_router_routes) - 1, -1, -1):
    _r = _router_routes[_i]
    if getattr(_r, "path", None) in {"/reset", "/step"} and getattr(_r, "methods", None):
        _router_routes.pop(_i)


async def _http_reset(request: Request) -> JSONResponse:
    env = _get_or_create_http_env(request)
    obs = env.reset()
    session_id = _http_session_key(request)
    return JSONResponse({
        "session_id": session_id,
        "observation": obs.model_dump(exclude_none=True, mode="json"),
        "reward": obs.reward,
        "done": obs.done,
    })


async def _http_step(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"detail": "request body must be JSON"}, status_code=400)

    action_payload = payload.get("action") if isinstance(payload, dict) else None
    if not isinstance(action_payload, dict):
        return JSONResponse({"detail": "action must be a JSON object"}, status_code=400)

    try:
        action = FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return JSONResponse({"detail": f"invalid action: {exc}"}, status_code=422)

    env = _get_or_create_http_env(request)
    obs = env.step(action)
    return JSONResponse({
        "observation": obs.model_dump(exclude_none=True, mode="json"),
        "reward": obs.reward,
        "done": obs.done,
    })


from starlette.routing import Route  # noqa: E402

app.router.routes.insert(0, Route("/reset", _http_reset, methods=["POST"]))
app.router.routes.insert(1, Route("/step", _http_step, methods=["POST"]))


# ── CORS allowlist (env-driven) ──────────────────────────────────────────────
# Defaults are dev-friendly; tighten via ALLOWED_ORIGINS="https://a.com,https://b.com".

_allowed_origins = config.allowed_origins()
_allow_credentials = "*" not in _allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── API-key authentication (env-driven) ──────────────────────────────────────
# Cache keys at module-load time. To rotate, restart the server.
# Public prefixes (health, docs, dashboard, metrics SSE) bypass auth so the
# UI/operators can probe the service without a key.

_PUBLIC_PREFIXES = config.PUBLIC_ROUTE_PREFIXES


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


class SeedScopeMiddleware(BaseHTTPMiddleware):
    """Binds request-specific seed scope for non-global reset/eval isolation."""

    async def dispatch(self, request: Request, call_next):
        if request.scope.get("type") == "websocket":
            return await call_next(request)
        token = _REQUEST_SEED_SCOPE.set(_request_seed_scope(request))
        try:
            return await call_next(request)
        finally:
            _REQUEST_SEED_SCOPE.reset(token)


app.add_middleware(SeedScopeMiddleware)


_RATE_LIMIT_RPS = float(_os.environ.get("FRAUD_HUNTER_RATE_LIMIT_RPS", "8"))
_RATE_LIMIT_BURST = float(_os.environ.get("FRAUD_HUNTER_RATE_LIMIT_BURST", "16"))

_AGENT_MODEL = (
    (_os.environ.get("FRAUD_HUNTER_LLM_MODEL") or "").strip()
    or (_os.environ.get("OPENAI_MODEL") or "").strip()
)
_AGENT_BASE_URL = (
    (_os.environ.get("FRAUD_HUNTER_LLM_BASE_URL") or "").strip().rstrip("/")
    or (_os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
)
_AGENT_API_KEY = (
    (_os.environ.get("FRAUD_HUNTER_LLM_API_KEY") or "").strip()
    or (_os.environ.get("OPENAI_API_KEY") or "").strip()
)
_AGENT_EXPLICIT = (_os.environ.get("FRAUD_HUNTER_AGENT_ENABLED") or "").strip().lower()
_AGENT_ENABLED = (
    _AGENT_EXPLICIT in {"1", "true", "yes", "on"}
    if _AGENT_EXPLICIT
    else bool(_AGENT_MODEL and _AGENT_BASE_URL)
)

_ONLINE_RL_EXPLICIT = (_os.environ.get("FRAUD_HUNTER_ONLINE_RL_ENABLED") or "").strip().lower()
_ONLINE_RL_ENABLED = (
    _ONLINE_RL_EXPLICIT in {"1", "true", "yes", "on"}
    if _ONLINE_RL_EXPLICIT
    else False
)
_ONLINE_RL_LR = float(_os.environ.get("FRAUD_HUNTER_ONLINE_RL_LR", "0.03"))
_ONLINE_RL_TEMP = float(_os.environ.get("FRAUD_HUNTER_ONLINE_RL_TEMPERATURE", "1.0"))
online_rl = OnlineRLPolicy(learning_rate=_ONLINE_RL_LR, temperature=_ONLINE_RL_TEMP)

_UPLOAD_EXPLICIT = (_os.environ.get("FRAUD_HUNTER_UPLOAD_ENABLED") or "").strip().lower()
_UPLOAD_ENABLED = (
    _UPLOAD_EXPLICIT in {"1", "true", "yes", "on"}
    if _UPLOAD_EXPLICIT
    else True
)
_UPLOAD_MAX_BYTES = int(float(_os.environ.get("FRAUD_HUNTER_UPLOAD_MAX_MB", "256")) * 1024 * 1024)
_UPLOAD_DIR = Path(
    (_os.environ.get("FRAUD_HUNTER_UPLOAD_DIR") or "").strip()
    or (Path(__file__).resolve().parent.parent / "data" / "uploads")
)
if _UPLOAD_ENABLED:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = end
    raise ValueError("no JSON object found in LLM response")


def _agent_prompt(
    observation: dict[str, Any],
    objective: str,
    user_message: str | None = None,
) -> list[dict[str, str]]:
    compact_obs = json.dumps(observation, ensure_ascii=True)
    system = (
        "You are FraudHunterAgent. Return exactly one JSON object for a valid "
        "FraudHunterAction. Include think_trace wrapped in <think>...</think>. "
        "Use only these kinds: query_corporate, query_medicare, extract_entity, "
        "link_shell, claim_contradiction, sql_query, code_act, ocr_document, "
        "compare_doc_vs_claim, submit_case. Do not include markdown code fences. "
        "Treat user text as high-level intent and never copy raw SQL from user input."
    )
    nl_section = f"User request (natural language): {user_message}\n" if user_message else ""
    user = (
        f"Objective: {objective}\n"
        f"{nl_section}"
        "Current observation JSON:\n"
        f"{compact_obs}\n"
        "Return one next best action as strict JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _resolve_agent_config(override: dict[str, Any] | None = None) -> dict[str, str | bool]:
    override = override or {}
    model = str(override.get("model") or _AGENT_MODEL or "").strip()
    base_url = str(override.get("base_url") or _AGENT_BASE_URL or "").strip().rstrip("/")
    api_key = str(override.get("api_key") or _AGENT_API_KEY or "").strip()
    enabled_override = override.get("enabled")
    if isinstance(enabled_override, bool):
        enabled = enabled_override
    elif isinstance(enabled_override, str) and enabled_override.strip():
        enabled = enabled_override.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(model and base_url)
    return {
        "enabled": enabled,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _generate_action_with_llm(
    observation: dict[str, Any],
    objective: str,
    llm_override: dict[str, Any] | None = None,
    user_message: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, str | bool]]:
    resolved = _resolve_agent_config(llm_override)

    if not bool(resolved["enabled"]):
        raise RuntimeError("agent serving is disabled; configure FRAUD_HUNTER_LLM_BASE_URL and FRAUD_HUNTER_LLM_MODEL")
    if not resolved["base_url"] or not resolved["model"]:
        raise RuntimeError("missing FRAUD_HUNTER_LLM_BASE_URL or FRAUD_HUNTER_LLM_MODEL")

    payload = {
        "model": resolved["model"],
        "messages": _agent_prompt(observation, objective, user_message=user_message),
        "temperature": 0.2,
        "max_tokens": 380,
    }
    req = urllib.request.Request(
        f"{resolved['base_url']}/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    if resolved["api_key"]:
        req.add_header("Authorization", f"Bearer {resolved['api_key']}")

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
        content = (((parsed.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except Exception as exc:
        raise RuntimeError(f"invalid LLM response payload: {exc}") from exc

    action_payload = _extract_first_json_object(str(content))
    return action_payload, str(content), resolved


def _heuristic_action(observation: dict[str, Any], objective: str) -> dict[str, Any]:
    """Fallback action when LLM arm is selected but LLM is unavailable."""
    return online_rl.fallback_action(observation, objective)


def _safe_dataset_name(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (raw or "").strip())
    return value[:120] or f"dataset_{int(time.time())}"


def _safe_join(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    base_resolved = base.resolve()
    if base_resolved == candidate or base_resolved in candidate.parents:
        return candidate
    raise ValueError("invalid archive path traversal attempt")


async def _save_upload_file(upload: UploadFile, destination: Path) -> int:
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > _UPLOAD_MAX_BYTES:
                raise ValueError(f"file exceeds max upload size {_UPLOAD_MAX_BYTES} bytes")
            f.write(chunk)
    return written


def _extract_zip_safe(zip_path: Path, target_dir: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.file_size > _UPLOAD_MAX_BYTES:
                raise ValueError("archive contains oversized file")
            relative = info.filename.replace("\\", "/")
            out_path = _safe_join(target_dir, relative)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            extracted.append(str(out_path.relative_to(target_dir)))
    return extracted


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket limiter to protect worker pools under concurrent load."""

    def __init__(self, app):
        super().__init__(app)
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def _key_for(self, request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    async def dispatch(self, request: Request, call_next):
        if request.scope.get("type") == "websocket" or request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if any(path == p or path.startswith(p + "/") or path == p for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        now = time.monotonic()
        key = self._key_for(request)
        with self._lock:
            tokens, last = self._buckets.get(key, (_RATE_LIMIT_BURST, now))
            elapsed = max(0.0, now - last)
            tokens = min(_RATE_LIMIT_BURST, tokens + elapsed * _RATE_LIMIT_RPS)

            if tokens < 1.0:
                retry_after = max(0.05, (1.0 - tokens) / max(_RATE_LIMIT_RPS, 0.1))
                jitter = random.uniform(0.0, 0.25)
                retry_after_jittered = retry_after + jitter
                self._buckets[key] = (tokens, now)
                return JSONResponse(
                    {
                        "detail": "Too Many Requests",
                        "retry_after_seconds": round(retry_after_jittered, 3),
                    },
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(retry_after_jittered)))},
                )

            self._buckets[key] = (tokens - 1.0, now)

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ── Startup warnings ──────────────────────────────────────────────────────────
import logging as _logging

_startup_logger = _logging.getLogger("fraud_hunter_env.app")
if not _API_KEYS:
    _startup_logger.warning(
        "FRAUD_HUNTER_API_KEYS not set — API key auth is DISABLED (open sandbox mode)"
    )
if _ONLINE_RL_ENABLED:
    _startup_logger.info(
        "Online RL is ENABLED (FRAUD_HUNTER_ONLINE_RL_ENABLED=true); "
        "policy weights will update on every episode terminal reward."
    )


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

    def _dashboard_runtime_config(request: Request) -> dict[str, Any]:
        """Expose runtime UI wiring so the dashboard can auto-configure itself."""
        host = request.headers.get("host", "localhost:8000")
        ws_scheme = "wss" if request.url.scheme == "https" else "ws"
        return {
            "name": "fraud_hunter_env",
            "auth_required": bool(_API_KEYS),
            "headers": {
                "api_key": "x-api-key",
                "seed_scope": "x-seed-scope",
            },
            "endpoints": {
                "health": "/health",
                "reset": "/reset",
                "step": "/step",
                "session_reset": "/fraud_hunter/session_reset",
                "session_step": "/fraud_hunter/session_step",
                "agent_action": "/fraud_hunter/agent_action",
                "agent_action_online": "/fraud_hunter/agent_action_online",
                "nl_action": "/fraud_hunter/nl_action",
                "online_rl_update": "/fraud_hunter/online_rl/update",
                "online_rl_state": "/fraud_hunter/online_rl/state",
                "online_rl_reset": "/fraud_hunter/online_rl/reset",
                "upload_dataset": "/fraud_hunter/upload_dataset",
                "state": "/state",
                "metrics": "/metrics",
                "metrics_history": "/metrics/history",
                "ws": "/ws",
                "fraud_health": "/fraud_hunter/health",
            },
            "agent": {
                **_resolve_agent_config(),
                "api_key": "",
            },
            "online_rl": {
                "enabled": _ONLINE_RL_ENABLED,
                "learning_rate": _ONLINE_RL_LR,
                "temperature": _ONLINE_RL_TEMP,
            },
            "upload": {
                "enabled": _UPLOAD_ENABLED,
                "max_bytes": _UPLOAD_MAX_BYTES,
            },
            "urls": {
                "http_base": str(request.base_url).rstrip("/"),
                "ws": f"{ws_scheme}://{host}/ws",
            },
        }

    @app.get("/dashboard/config", include_in_schema=False)
    async def dashboard_config(request: Request):
        # Note: this path may be shadowed by the /dashboard StaticFiles mount.
        return _dashboard_runtime_config(request)

    @app.get("/ui/config", include_in_schema=False)
    async def ui_config(request: Request):
        # Stable config endpoint for frontend auto-wiring.
        return _dashboard_runtime_config(request)
else:
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_missing():
        return HTMLResponse(
            "<h1>Dashboard not found. web/index.html is missing.</h1>",
            status_code=404,
        )


# ── Root redirect → dashboard ─────────────────────────────────────────────────
# HF Spaces users land at / — redirect them straight to the UI.

from fastapi.responses import RedirectResponse  # noqa: E402 (already imported above)

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect bare root to the custom dashboard UI."""
    return RedirectResponse(url="/dashboard/", status_code=302)

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
    active_scope = _REQUEST_SEED_SCOPE.get()
    return {
        "status": "healthy",
        "episodes_logged": metrics_bus.episode_count(),
        "sse_clients": metrics_bus.subscriber_count(),
        "seed_scope": active_scope,
        "seed_range": _active_seed_range(),
    }


@app.get("/fraud_hunter/seed_range")
async def get_seed_range(request: Request):
    scope = _request_seed_scope(request)
    with _SEED_SCOPE_LOCK:
        value = _SEED_RANGE_BY_SCOPE.get(scope)
    if value is None:
        return {"scope": scope, "seed_min": None, "seed_max": None}
    return {"scope": scope, "seed_min": value[0], "seed_max": value[1]}


@app.post("/fraud_hunter/seed_range")
async def set_seed_range(payload: dict[str, int | None], request: Request):
    seed_min = payload.get("seed_min")
    seed_max = payload.get("seed_max")
    if (seed_min is None) != (seed_max is None):
        return JSONResponse({"detail": "seed_min and seed_max must both be set or both be null"}, status_code=400)
    if seed_min is not None and seed_max is not None and seed_min > seed_max:
        return JSONResponse({"detail": "seed_min must be <= seed_max"}, status_code=400)

    scope = _request_seed_scope(request)
    with _SEED_SCOPE_LOCK:
        existing = _SEED_RANGE_BY_SCOPE.get(scope)
        next_value: tuple[int, int] | None = None
        if seed_min is not None and seed_max is not None:
            next_value = (int(seed_min), int(seed_max))

        # Immutable-by-default semantics: changing an already pinned range
        # requires an explicit clear (seed_min=null, seed_max=null) first.
        if existing is not None and next_value is not None and existing != next_value:
            return JSONResponse(
                {
                    "detail": (
                        "seed range for this scope is immutable once set; "
                        "clear it with null/null before changing"
                    )
                },
                status_code=409,
            )
        _SEED_RANGE_BY_SCOPE[scope] = next_value

    return {"scope": scope, "seed_min": seed_min, "seed_max": seed_max}


@app.post("/fraud_hunter/session_reset")
async def fraud_hunter_session_reset(request: Request):
    env = _get_or_create_http_env(request)
    obs = env.reset()
    return {
        "observation": obs.model_dump(exclude_none=True, mode="json"),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/fraud_hunter/session_step")
async def fraud_hunter_session_step(payload: dict[str, Any], request: Request):
    action_payload = payload.get("action")
    if not isinstance(action_payload, dict):
        return JSONResponse({"detail": "action must be a JSON object"}, status_code=400)

    try:
        action = FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return JSONResponse({"detail": f"invalid action payload: {exc}"}, status_code=422)

    env = _get_or_create_http_env(request)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(exclude_none=True, mode="json"),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/fraud_hunter/agent_action")
async def fraud_hunter_agent_action(payload: dict[str, Any]):
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        return JSONResponse({"detail": "observation must be a JSON object"}, status_code=400)
    objective = str(payload.get("objective") or "Investigate and produce the highest-value next action.")
    user_message = str(payload.get("user_message") or "").strip() or None

    llm_override = payload.get("llm")
    if llm_override is not None and not isinstance(llm_override, dict):
        return JSONResponse({"detail": "llm must be a JSON object when provided"}, status_code=400)

    try:
        action_payload, raw_content, resolved = _generate_action_with_llm(
            observation,
            objective,
            llm_override=llm_override if isinstance(llm_override, dict) else None,
            user_message=user_message,
        )
    except Exception as exc:
        return JSONResponse({"detail": f"agent generation failed: {exc}"}, status_code=503)

    try:
        action = FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return JSONResponse(
            {
                "detail": f"generated action failed schema validation: {exc}",
                "generated_action": action_payload,
                "raw_model_output": raw_content,
            },
            status_code=422,
        )

    return {
        "action": action.model_dump(exclude_none=True, mode="json"),
        "model": resolved["model"],
        "base_url": resolved["base_url"],
        "provider": "openai-compatible",
    }


@app.post("/fraud_hunter/nl_action")
async def fraud_hunter_nl_action(payload: dict[str, Any]):
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        return JSONResponse({"detail": "observation must be a JSON object"}, status_code=400)

    user_message = str(payload.get("user_message") or "").strip()
    if not user_message:
        return JSONResponse({"detail": "user_message is required"}, status_code=400)

    objective = str(payload.get("objective") or "Investigate and produce the highest-value next action.")
    llm_override = payload.get("llm")
    if llm_override is not None and not isinstance(llm_override, dict):
        return JSONResponse({"detail": "llm must be a JSON object when provided"}, status_code=400)

    used_fallback = False
    fallback_reason: str | None = None
    try:
        action_payload, raw_content, resolved = _generate_action_with_llm(
            observation,
            objective,
            llm_override=llm_override if isinstance(llm_override, dict) else None,
            user_message=user_message,
        )
    except Exception as exc:
        # Keep NL chat operational even when LLM config/provider is unavailable.
        used_fallback = True
        fallback_reason = str(exc)
        action_payload = _heuristic_action(observation, objective)
        raw_content = f"heuristic_fallback: {fallback_reason}"
        resolved = {
            "model": "heuristic",
            "base_url": "",
            "enabled": False,
            "api_key": "",
        }

    try:
        action = FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return JSONResponse(
            {
                "detail": f"generated action failed schema validation: {exc}",
                "generated_action": action_payload,
                "raw_model_output": raw_content,
            },
            status_code=422,
        )

    response: dict[str, Any] = {
        "action": action.model_dump(exclude_none=True, mode="json"),
        "model": resolved["model"],
        "base_url": resolved["base_url"],
        "provider": "heuristic-fallback" if used_fallback else "openai-compatible",
    }
    if used_fallback and fallback_reason:
        response["llm_error"] = fallback_reason
    return response


@app.post("/fraud_hunter/agent_action_online")
async def fraud_hunter_agent_action_online(payload: dict[str, Any]):
    if not _ONLINE_RL_ENABLED:
        return JSONResponse({"detail": "online RL is disabled"}, status_code=503)

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        return JSONResponse({"detail": "observation must be a JSON object"}, status_code=400)
    objective = str(payload.get("objective") or "Investigate and produce the highest-value next action.")
    user_message = str(payload.get("user_message") or "").strip() or None

    llm_override = payload.get("llm")
    if llm_override is not None and not isinstance(llm_override, dict):
        return JSONResponse({"detail": "llm must be a JSON object when provided"}, status_code=400)

    llm_cfg = _resolve_agent_config(llm_override if isinstance(llm_override, dict) else None)
    allow_llm = bool(llm_cfg["enabled"] and llm_cfg["base_url"] and llm_cfg["model"])

    decision = online_rl.choose_arm(observation, objective, allow_llm=allow_llm)
    arm = str(decision.get("arm") or "")
    token = str(decision.get("token") or "")
    probs = decision.get("probs") if isinstance(decision.get("probs"), dict) else {}

    action_payload: dict[str, Any]
    raw_model_output: str | None = None
    model = "heuristic"
    base_url = ""
    provider = "online-rl-template"

    if arm == "llm":
        try:
            action_payload, raw_model_output, resolved = _generate_action_with_llm(
                observation,
                objective,
                llm_override=llm_override if isinstance(llm_override, dict) else None,
                user_message=user_message,
            )
            model = str(resolved["model"] or "")
            base_url = str(resolved["base_url"] or "")
            provider = "openai-compatible"
        except Exception as exc:
            # Keep the online loop live even when the provider is unavailable.
            action_payload = _heuristic_action(observation, objective)
            raw_model_output = f"llm_fallback: {exc}"
            provider = "online-rl-fallback"
    else:
        maybe_action = decision.get("action")
        if not isinstance(maybe_action, dict):
            maybe_action = _heuristic_action(observation, objective)
        action_payload = maybe_action

    try:
        action = FraudHunterAction.model_validate(action_payload)
    except Exception as exc:
        return JSONResponse(
            {
                "detail": f"generated action failed schema validation: {exc}",
                "generated_action": action_payload,
                "raw_model_output": raw_model_output,
            },
            status_code=422,
        )

    return {
        "decision_token": token,
        "arm": arm,
        "arm_probs": probs,
        "action": action.model_dump(exclude_none=True, mode="json"),
        "model": model,
        "base_url": base_url,
        "provider": provider,
        "online_rl": online_rl.snapshot(),
    }


@app.post("/fraud_hunter/upload_dataset")
async def fraud_hunter_upload_dataset(
    file: list[UploadFile] = File(...),
    dataset_name: str | None = Form(default=None),
    extract_zip: bool = Form(default=True),
):
    if not _UPLOAD_ENABLED:
        return JSONResponse({"detail": "dataset upload is disabled"}, status_code=503)

    uploads = file
    if not uploads:
        return JSONResponse({"detail": "at least one file is required"}, status_code=400)

    first_name = (uploads[0].filename or "dataset").strip()
    safe_name = _safe_dataset_name(dataset_name or Path(first_name).stem)
    target_root = _UPLOAD_DIR / safe_name
    target_root.mkdir(parents=True, exist_ok=True)
    stored_files: list[str] = []
    extracted_files: list[str] = []
    content_types: list[str] = []
    total_bytes = 0
    extracted_any = False

    for idx, upload in enumerate(uploads):
        original_name = (upload.filename or f"dataset_{idx + 1}").strip()
        suffix = Path(original_name).suffix.lower()
        guessed_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        content_types.append(upload.content_type or guessed_type)

        if suffix not in {".csv", ".zip"}:
            return JSONResponse({"detail": "only .csv and .zip uploads are supported"}, status_code=400)

        stem = safe_name if len(uploads) == 1 else _safe_dataset_name(Path(original_name).stem)
        stored_path = target_root / f"{stem}{suffix}"
        if stored_path.exists() and len(uploads) > 1:
            stored_path = target_root / f"{stem}_{idx + 1}{suffix}"

        try:
            total_bytes += await _save_upload_file(upload, stored_path)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=413)
        finally:
            await upload.close()

        stored_files.append(str(stored_path.relative_to(_UPLOAD_DIR)))
        if suffix == ".zip" and extract_zip:
            try:
                extracted_files.extend(_extract_zip_safe(stored_path, target_root))
                extracted_any = True
            except Exception as exc:
                return JSONResponse({"detail": f"zip extraction failed: {exc}"}, status_code=400)

    return {
        "status": "uploaded",
        "dataset": safe_name,
        "stored_file": stored_files[0],
        "stored_files": stored_files,
        "content_type": content_types[0],
        "content_types": content_types,
        "bytes": total_bytes,
        "file_count": len(stored_files),
        "extract_zip": extracted_any,
        "extracted_count": len(extracted_files),
        "extracted_files": extracted_files[:200],
        "upload_root": str(_UPLOAD_DIR),
    }


@app.post("/fraud_hunter/online_rl/update")
async def fraud_hunter_online_rl_update(payload: dict[str, Any]):
    if not _ONLINE_RL_ENABLED:
        return JSONResponse({"detail": "online RL is disabled"}, status_code=503)

    token = str(payload.get("decision_token") or "").strip()
    if not token:
        return JSONResponse({"detail": "decision_token is required"}, status_code=400)

    raw_reward = payload.get("reward")
    if raw_reward is None:
        return JSONResponse({"detail": "reward is required"}, status_code=400)
    try:
        reward = float(raw_reward)
    except Exception:
        return JSONResponse({"detail": "reward must be numeric"}, status_code=400)
    if not math.isfinite(reward):
        return JSONResponse({"detail": "reward must be finite"}, status_code=400)

    try:
        result = online_rl.update(token, reward)
    except KeyError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    return {
        "status": "updated",
        **result,
        "online_rl": online_rl.snapshot(),
    }


@app.get("/fraud_hunter/online_rl/state")
async def fraud_hunter_online_rl_state():
    return {
        "enabled": _ONLINE_RL_ENABLED,
        "state": online_rl.snapshot(),
    }


@app.post("/fraud_hunter/online_rl/reset")
async def fraud_hunter_online_rl_reset():
    if not _ONLINE_RL_ENABLED:
        return JSONResponse({"detail": "online RL is disabled"}, status_code=503)
    state = online_rl.reset()
    return {"status": "reset", "state": state}


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
