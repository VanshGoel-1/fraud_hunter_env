"""Backward-compatible entrypoint.

The demo implementation now lives in scripts/http_surface_check.py and
scripts/validate_runtime.py. Keep this file so existing docs/commands don't break.
"""

from scripts.http_surface_check import main


if __name__ == "__main__":
    raise SystemExit(main())
