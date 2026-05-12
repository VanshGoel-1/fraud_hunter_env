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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_CASE_BANK_DIR: Path = REPO_ROOT / "data" / "case_bank"


def case_bank_dir() -> Path:
    """Resolved case-bank directory (env override: ``FRAUD_HUNTER_CASE_BANK``)."""
    raw = os.environ.get("FRAUD_HUNTER_CASE_BANK")
    return Path(raw) if raw else DEFAULT_CASE_BANK_DIR


# ── CORS ─────────────────────────────────────────────────────────────────────

_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


@dataclass(frozen=True)
class CORSConfig:
    """Encapsulates CORS policy and registers it onto a FastAPI app.

    Resolved from the environment via ``CORSConfig.from_env()``.
    Env var ``ALLOWED_ORIGINS`` accepts a comma-separated list of origins.
    Pass ``*`` to allow all origins (open; only suitable for public read-only
    deployments or local development where auth is handled separately).
    """

    origins: list[str] = field(default_factory=lambda: list(_DEFAULT_ORIGINS))
    allow_credentials: bool = True
    methods: list[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    headers: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "CORSConfig":
        raw = (os.environ.get("ALLOWED_ORIGINS") or "").strip()
        if raw:
            origins = [o.strip() for o in raw.split(",") if o.strip()]
        elif os.environ.get("SPACE_ID"):
            # HF Spaces: the Playground UI is served from huggingface.co
            origins = ["*"]
        else:
            origins = list(_DEFAULT_ORIGINS)
        return cls(
            origins=origins,
            # credentials cookies are incompatible with wildcard origin
            allow_credentials="*" not in origins,
        )

    def register(self, app: "FastAPI") -> None:
        """Attach CORSMiddleware to *app* using this config."""
        from fastapi.middleware.cors import CORSMiddleware  # runtime import only

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.origins,
            allow_credentials=self.allow_credentials,
            allow_methods=self.methods,
            allow_headers=self.headers,
        )


# ── HTTP / auth ──────────────────────────────────────────────────────────────

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
