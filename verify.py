"""Compiler-in-the-loop: cheap static checks run on files the agent just wrote.

Nothing here executes the file's code. Python is compiled to bytecode in
memory (never imported or run), JSON/YAML are parsed, and JavaScript is
checked with `node --check`, which parses without running.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

MAX_ERROR_CHARS = 1500


def check_file(path: Path) -> Optional[str]:
    """Return a short error description if `path` fails its checker, else None.

    Unknown extensions and unreadable files return None: no checker means
    nothing to report, not a failure.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            return _check_python(path)
        if suffix == ".json":
            return _check_json(path)
        if suffix in {".yaml", ".yml"}:
            return _check_yaml(path)
        if suffix in {".js", ".mjs", ".cjs"}:
            return _check_node(path)
    except OSError:
        return None
    return None


def _check_python(path: Path) -> Optional[str]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(source, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        line = (exc.text or "").rstrip()
        where = f"line {exc.lineno}" if exc.lineno else "unknown line"
        detail = f"SyntaxError: {exc.msg} ({where})"
        return f"{detail}\n    {line}" if line else detail
    return None


def _check_json(path: Path) -> Optional[str]:
    try:
        json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return f"JSON error: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    return None


def _check_yaml(path: Path) -> Optional[str]:
    try:
        import yaml
    except ImportError:
        return None
    try:
        yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except yaml.YAMLError as exc:
        return f"YAML error: {str(exc)[:MAX_ERROR_CHARS]}"
    return None


def _check_node(path: Path) -> Optional[str]:
    node = shutil.which("node")
    if not node:
        return None
    try:
        proc = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip()[:MAX_ERROR_CHARS] or "JavaScript syntax error"
    return None
