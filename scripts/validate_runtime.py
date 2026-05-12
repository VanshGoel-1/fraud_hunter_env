"""Runtime validation entrypoint without pytest.

Runs fast compile checks and the HTTP action-surface probe.
"""

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compile_targets() -> None:
    targets = [
        PROJECT_ROOT / "models.py",
        PROJECT_ROOT / "client.py",
        PROJECT_ROOT / "server" / "app.py",
        PROJECT_ROOT / "server" / "sandbox.py",
        PROJECT_ROOT / "server" / "grader.py",
        PROJECT_ROOT / "server" / "fraud_hunter_env_environment.py",
        PROJECT_ROOT / "server" / "difficulty.py",
        PROJECT_ROOT / "eval.py",
        PROJECT_ROOT / "demo.py",
        PROJECT_ROOT / "scripts" / "http_surface_check.py",
    ]
    for target in targets:
        py_compile.compile(str(target), doraise=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--seed-min", type=int, default=8001)
    parser.add_argument("--seed-max", type=int, default=10000)
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument("--remote-http", action="store_true")
    args = parser.parse_args()

    _compile_targets()

    if not args.skip_http:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "http_surface_check.py"),
            "--seed-min",
            str(args.seed_min),
            "--seed-max",
            str(args.seed_max),
        ]
        if args.remote_http:
            cmd.extend(["--remote", "--base-url", args.base_url])
        subprocess.run(cmd, check=True)

    print("runtime validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
