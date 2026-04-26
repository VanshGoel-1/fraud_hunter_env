"""
Centralized runtime configuration.

All environment-variable-driven knobs live here so callers don't reach into
`os.environ` from across the codebase. Values are resolved at import time
(once per process) — to rotate them, restart the server.

Why not pydantic-settings: keeping zero extra deps. Drop-in upgrade if/when
we need nested config or per-environment overlays.
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_CASE_BANK_DIR: Path = REPO_ROOT / "data" / "case_bank"


def case_bank_dir() -> Path:
    """Resolved case-bank directory (env override: ``FRAUD_HUNTER_CASE_BANK``)."""
    raw = os.environ.get("FRAUD_HUNTER_CASE_BANK")
    return Path(raw) if raw else DEFAULT_CASE_BANK_DIR


# ── HTTP / auth ──────────────────────────────────────────────────────────────

_DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
)


def allowed_origins() -> list[str]:
    """CORS allowlist (env: ``ALLOWED_ORIGINS`` — comma-separated).

    Defaults to ``*`` on HF Spaces (detected via ``SPACE_ID``) so the
    built-in Playground UI — served from huggingface.co — can reach the
    server.  Override with ``ALLOWED_ORIGINS`` to tighten in production.
    """
    raw = os.environ.get("ALLOWED_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if os.environ.get("SPACE_ID"):
        return ["*"]
    return [o.strip() for o in _DEFAULT_ALLOWED_ORIGINS.split(",") if o.strip()]


def api_keys() -> set[str]:
    """Authorized X-API-Key set (env: ``FRAUD_HUNTER_API_KEYS`` — comma-separated).

    Empty set ⇒ auth disabled (dev mode).
    """
    raw = os.environ.get("FRAUD_HUNTER_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


# Public route prefixes that bypass APIKeyMiddleware.
PUBLIC_ROUTE_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/dashboard",
    "/ui",
    "/assets",
    "/metrics",
    "/fraud_hunter/health",
)


# ── Server ───────────────────────────────────────────────────────────────────

def server_host() -> str:
    return os.environ.get("FRAUD_HUNTER_HOST", "0.0.0.0")


def server_port() -> int:
    return int(os.environ.get("FRAUD_HUNTER_PORT", "8000"))
