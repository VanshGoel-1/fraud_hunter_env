"""
CodeAct sandbox for the Fraud Hunter environment.

Allows agents to submit Python code that is executed in a restricted,
time-limited sandbox with read-only access to the case SQLite database.

Security boundaries:
  - Restricted builtins (no open/exec/import/eval/os/sys/subprocess)
  - 5-second timeout enforced via threading
  - No network I/O, no filesystem writes
  - Pre-injected: conn (read-only sqlite3), pd (pandas), json
  - Output capped at 4096 characters

This implements the CodeAct paradigm where the agent can write arbitrary
SQL-backed Python to extract, join and assert data in a single step,
earning CODEACT_BONUS per successful DataFrame row returned.
"""

from __future__ import annotations

import ast
import io
import json
import os
import sqlite3
import threading
import traceback
from contextlib import redirect_stdout
from typing import Optional

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "dir": dir, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset,
    "hasattr": hasattr, "hash": hash, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

# AST-based safety: reject these node-name attribute targets and these names.
_FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "ctypes", "importlib", "builtins", "__builtins__", "__import__",
    "multiprocessing", "threading",
})
_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint",
    "__builtins__", "__class__", "__bases__", "__subclasses__",
    "__mro__", "__globals__", "__getattribute__",
})
_FORBIDDEN_DUNDERS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__getattribute__", "__reduce__", "__reduce_ex__",
    "__init_subclass__", "__import__",
})

# Belt-and-suspenders substring blocklist; runs after the AST check.
_FORBIDDEN_SUBSTRINGS = (
    "__import__", "__builtins__", "__class__", "__subclasses__",
    "__bases__", "__mro__", "__globals__",
)

_MAX_OUTPUT_CHARS = 4096
_TIMEOUT_SECONDS = 5


class _SafetyVisitor(ast.NodeVisitor):
    """Walks the AST and records the first safety violation, if any."""

    def __init__(self) -> None:
        self.violation: Optional[str] = None

    def _flag(self, msg: str) -> None:
        if self.violation is None:
            self.violation = msg

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(a.name for a in node.names)
        self._flag(f"import statement disallowed: {names}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._flag(f"from-import disallowed: from {node.module or '?'}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self._flag(f"forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Reject any access on a forbidden module name (e.g. os.system).
        target = node
        while isinstance(target, ast.Attribute):
            target = target.value
        if isinstance(target, ast.Name) and target.id in _FORBIDDEN_MODULES:
            self._flag(f"forbidden attribute access on: {target.id}")
        if node.attr in _FORBIDDEN_DUNDERS:
            self._flag(f"forbidden dunder attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Reject getattr/setattr/delattr — they smuggle attribute access.
        if isinstance(node.func, ast.Name) and node.func.id in {
            "getattr", "setattr", "delattr"
        }:
            self._flag(f"forbidden call: {node.func.id}()")
        self.generic_visit(node)


def _check_code_safety(code: str) -> Optional[str]:
    """Returns an error message if the code is unsafe, None if safe."""
    # Layer 1: cheap substring scan catches the most obvious dunder smuggling.
    for needle in _FORBIDDEN_SUBSTRINGS:
        if needle in code:
            return f"Forbidden pattern detected: {needle!r}"
    # Layer 2: AST walk catches structural attempts (import, getattr, etc.).
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    if visitor.violation:
        return visitor.violation
    return None


def execute_code(
    code: str,
    conn: sqlite3.Connection,
    case_dir: Optional[str] = None,
) -> tuple[str, Optional[str], int]:
    """
    Execute `code` in a restricted sandbox with access to `conn`.

    Returns:
        (stdout_output, error_message, rows_returned)
        - stdout_output: captured print() output, capped at _MAX_OUTPUT_CHARS
        - error_message: None on success, traceback string on error
        - rows_returned: number of DataFrame rows detected in output (for bonus scoring)
    """
    # Note: we deliberately do NOT call os.chdir here. The server is
    # multi-tenant and changing CWD races across concurrent sessions; all
    # filesystem helpers below take `case_dir` as the explicit base.

    safety_err = _check_code_safety(code)
    if safety_err:
        return "", f"SECURITY_VIOLATION: {safety_err}", 0

    # Inject PDF dependencies if available
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None
    try:
        import pytesseract
    except ImportError:
        pytesseract = None
    try:
        from PIL import Image
    except ImportError:
        Image = None

    # Narrow filesystem helpers: only within the case directory subtree. This
    # lets the agent enumerate intercepted_comms/ and scanned_claims/ without
    # needing `import os` (which the forbidden-pattern scanner would reject).
    _case_base = os.path.abspath(case_dir) if case_dir else None

    def _confined(target: str) -> str:
        if _case_base is None:
            raise PermissionError("filesystem access requires a case_dir")
        resolved = os.path.abspath(
            target if os.path.isabs(target) else os.path.join(_case_base, target)
        )
        if not (resolved == _case_base or resolved.startswith(_case_base + os.sep)):
            raise PermissionError(f"path escapes case directory: {target!r}")
        return resolved

    def _safe_listdir(subdir: str = ".") -> list[str]:
        return sorted(os.listdir(_confined(subdir)))

    def _safe_path_join(*parts: str) -> str:
        # Anchor relative compositions at case_dir so paths work without
        # depending on the process CWD (which we deliberately do not change).
        if not parts:
            return _case_base or ""
        first = parts[0]
        if os.path.isabs(first) or _case_base is None:
            return os.path.join(*parts)
        return os.path.join(_case_base, *parts)

    def _safe_path_exists(path: str) -> bool:
        try:
            return os.path.exists(_confined(path))
        except PermissionError:
            return False

    def _safe_open(path: str, mode: str = "r", *args, **kwargs):
        # Read-only access, confined to case_dir. No append/write modes.
        if any(c in mode for c in ("w", "a", "+", "x")):
            raise PermissionError(f"write modes disallowed in sandbox: {mode!r}")
        return open(_confined(path), mode, *args, **kwargs)  # noqa: SIM115

    # Pre-injected stdlib modules — pure-python, no I/O, agent doesn't need to
    # import them (and import statements are now rejected by the AST check).
    import re as _re
    import datetime as _datetime
    import math as _math

    # Build restricted execution namespace
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "conn": conn,
        "json": json,
        "re": _re,
        "datetime": _datetime,
        "math": _math,
        "pdfplumber": pdfplumber,
        "pytesseract": pytesseract,
        "Image": Image,
        "open": _safe_open,               # Read-only, case-dir-confined
        "listdir": _safe_listdir,         # Enumerate evidence dirs
        "path_join": _safe_path_join,     # Compose relative paths
        "path_exists": _safe_path_exists, # Test evidence paths
        "result": None,
    }
    if _HAS_PANDAS:
        namespace["pd"] = pd

    stdout_capture = io.StringIO()
    error: Optional[str] = None
    rows_returned = 0

    def _run() -> None:
        nonlocal error, rows_returned
        try:
            with redirect_stdout(stdout_capture):
                exec(code, namespace)  # noqa: S102
            # Count rows if agent assigned a DataFrame to `result`
            if _HAS_PANDAS and isinstance(namespace.get("result"), pd.DataFrame):
                rows_returned = len(namespace["result"])
                with redirect_stdout(stdout_capture):
                    print(namespace["result"].to_string(max_rows=20))
        except Exception:
            error = traceback.format_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_TIMEOUT_SECONDS)

    if thread.is_alive():
        return "", f"TIMEOUT: Code exceeded {_TIMEOUT_SECONDS}s limit", 0

    output = stdout_capture.getvalue()
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    return output, error, rows_returned


def execute_sql(
    sql: str,
    conn: sqlite3.Connection,
    max_rows: int = 50,
) -> tuple[str, Optional[str], int]:
    """
    Execute a restricted SQL SELECT statement directly.
    Returns (formatted_result, error_message, rows_returned).
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "", "Only SELECT statements are permitted", 0

    # Block dangerous SQL patterns
    forbidden_sql = ["DROP ", "DELETE ", "INSERT ", "UPDATE ", "ATTACH ", "PRAGMA "]
    for pat in forbidden_sql:
        if pat in sql_upper:
            return "", f"Forbidden SQL operation: {pat.strip()}", 0

    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description] if cur.description else []
        if not rows:
            return "Query returned 0 rows.", None, 0
        lines = ["\t".join(cols)]
        for row in rows:
            lines.append("\t".join(str(v) for v in row))
        return "\n".join(lines), None, len(rows)
    except Exception as e:
        return "", f"SQL_ERROR: {e}", 0
