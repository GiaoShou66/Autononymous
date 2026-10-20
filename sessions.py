from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SESSIONS_DIR = ".agent_sessions"


def _atomic_replace(temporary_path: Path, target_path: Path, retries: int = 5) -> None:
    """
    temporary_path.replace(target_path), retrying briefly on a transient
    Windows file lock (WinError 5 / PermissionError) — e.g. antivirus
    scanning the target the instant it's written, or two processes (two
    webui.py instances, or cli.py + webui.py) autosaving the same session
    at nearly the same moment. Re-raises if it still fails after retries.
    """
    delay = 0.05
    for attempt in range(retries):
        try:
            temporary_path.replace(target_path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


class SessionStore:
    """
    Save and resume agent conversations to disk (Claude Code-style sessions).

    Each session is a JSON file under ``sessions_dir`` holding the agent's
    history, mode, and task list so a conversation can be picked up later.
    """

    def __init__(self, sessions_dir: Path | str) -> None:
        self.dir = Path(sessions_dir).expanduser().resolve()

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_.")
        if not safe:
            safe = time.strftime("%Y%m%d-%H%M%S")
        return self.dir / f"{safe}.json"

    def save(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        mode: str,
        tasks: List[Dict[str, Any]] | None = None,
        title: str = "New chat",
        plan_mode: bool = False,
        require_approval: bool = False,
        model_name: str = "",
    ) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        payload = {
            "id": path.stem,
            "saved": time.time(),
            "mode": mode,
            "model_name": model_name,
            "title": title.strip() or "New chat",
            "tasks": tasks or [],
            "plan_mode": plan_mode,
            "require_approval": require_approval,
            "history": history,
        }
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _atomic_replace(temporary_path, path)
        return path

    def load(self, session_id: str) -> Dict[str, Any]:
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"No session named '{session_id}' in {self.dir}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Corrupt session file.")
        return data

    def list_sessions(self) -> List[Dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        out: List[Dict[str, Any]] = []
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            history = data.get("history") or []
            out.append(
                {
                    "id": data.get("id", path.stem),
                    "saved": data.get("saved", path.stat().st_mtime),
                    "mode": data.get("mode", "?"),
                    "model_name": data.get("model_name", ""),
                    "title": data.get("title", "New chat"),
                    "messages": len(history),
                }
            )
        return out

    def rename(self, session_id: str, title: str) -> None:
        data = self.load(session_id)
        data["title"] = title.strip() or "New chat"
        data["saved"] = time.time()
        path = self._path(session_id)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _atomic_replace(temporary_path, path)

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True
