from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SETTINGS_DIR = ".agent"
SETTINGS_FILE = "settings.json"
HOOK_EVENTS = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop")


def settings_path(project_root: Path | str) -> Path:
    return Path(project_root) / SETTINGS_DIR / SETTINGS_FILE


def load_settings(project_root: Path | str) -> Dict[str, Any]:
    """Read .agent/settings.json; a missing or broken file yields empty settings."""
    path = settings_path(project_root)
    data: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}
    if not isinstance(data.get("hooks"), dict):
        data["hooks"] = {}
    return data


def save_settings(project_root: Path | str, data: Dict[str, Any]) -> Path:
    path = settings_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


@dataclass
class HookOutcome:
    blocked: bool = False
    # stderr of the hook that blocked (exit code 2), shown to the model.
    message: str = ""
    # stdout of hooks that exited 0; used as extra context for UserPromptSubmit.
    context: str = ""
    warnings: List[str] = field(default_factory=list)


class HookManager:
    """
    Runs user-defined shell commands at agent lifecycle events.

    Config shape (same events as Claude Code):
        {"PreToolUse": [{"matcher": "run_command|write_file", "command": "python check.py"}]}
    Claude Code's nested form {"matcher": ..., "hooks": [{"type": "command",
    "command": ...}]} is accepted too.

    A hook receives a JSON payload on stdin. Exit code 0 means ok (stdout is
    kept as context), exit code 2 blocks (stderr goes back to the model), any
    other exit code is a non-blocking warning.
    """

    def __init__(
        self,
        hooks: Optional[Dict[str, Any]],
        cwd: Path | str,
        timeout: int = 30,
    ) -> None:
        self.cwd = Path(cwd)
        self.timeout = timeout
        self.hooks: Dict[str, List[Dict[str, str]]] = {}
        for event, entries in (hooks or {}).items():
            if event in HOOK_EVENTS and isinstance(entries, list):
                self.hooks[event] = self._flatten(entries)

    @staticmethod
    def _flatten(entries: List[Any]) -> List[Dict[str, str]]:
        flat: List[Dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            matcher = str(entry.get("matcher") or "")
            if isinstance(entry.get("command"), str) and entry["command"].strip():
                flat.append({"matcher": matcher, "command": entry["command"]})
            for nested in entry.get("hooks") or []:
                if isinstance(nested, dict) and isinstance(nested.get("command"), str):
                    flat.append({"matcher": matcher, "command": nested["command"]})
        return flat

    def has(self, event: str) -> bool:
        return bool(self.hooks.get(event))

    def describe(self) -> List[str]:
        lines: List[str] = []
        for event in HOOK_EVENTS:
            for hook in self.hooks.get(event, []):
                matcher = hook["matcher"] or "*"
                lines.append(f"{event} [{matcher}]: {hook['command']}")
        return lines

    @staticmethod
    def _matches(matcher: str, tool_name: str) -> bool:
        if matcher in {"", "*"}:
            return True
        try:
            return re.fullmatch(matcher, tool_name) is not None
        except re.error:
            return matcher == tool_name

    def run(self, event: str, payload: Dict[str, Any], tool_name: str = "") -> HookOutcome:
        outcome = HookOutcome()
        body = dict(payload)
        body["hook_event_name"] = event
        body["cwd"] = str(self.cwd)
        stdin_text = json.dumps(body, ensure_ascii=False)
        env = {**os.environ, "AGENT_PROJECT_DIR": str(self.cwd)}

        for hook in self.hooks.get(event, []):
            if tool_name and not self._matches(hook["matcher"], tool_name):
                continue
            try:
                proc = subprocess.run(
                    hook["command"],
                    shell=True,
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=str(self.cwd),
                    env=env,
                )
            except subprocess.TimeoutExpired:
                outcome.warnings.append(f"hook timed out after {self.timeout}s: {hook['command']}")
                continue
            except OSError as exc:
                outcome.warnings.append(f"hook failed to start ({exc}): {hook['command']}")
                continue

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode == 0:
                if stdout:
                    outcome.context = (outcome.context + "\n" + stdout).strip()
            elif proc.returncode == 2:
                outcome.blocked = True
                outcome.message = stderr or stdout or f"Blocked by hook: {hook['command']}"
                return outcome
            else:
                outcome.warnings.append(
                    f"hook exited {proc.returncode}: {hook['command']}"
                    + (f" ({stderr})" if stderr else "")
                )
        return outcome
