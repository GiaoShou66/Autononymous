from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


DEFAULT_MEMORY_FILE = ".agent_memory.json"
DEFAULT_MAX_LESSONS = 200


@dataclass
class AgentMemory:
    """
    Persistent long-term memory for the agent.

    Two kinds of memory are stored in a single JSON file so they survive
    process restarts:

    - ``lessons``: durable, self-authored notes the agent chooses to keep
      (preferences, facts about the project, mistakes not to repeat). This is
      what lets the agent "self-improve" across sessions.
    - ``rolling_summary``: a compressed summary of older conversation turns
      that were dropped from the live context window. Instead of discarding
      old history outright, the agent folds it into this summary, so the gist
      of a long session is retained without keeping every message.
    """

    file_path: Path
    lessons: List[str] = field(default_factory=list)
    rolling_summary: str = ""
    max_lessons: int = DEFAULT_MAX_LESSONS
    updated: float = 0.0

    @classmethod
    def load(cls, file_path: Path | str, max_lessons: int = DEFAULT_MAX_LESSONS) -> "AgentMemory":
        path = Path(file_path).expanduser().resolve()
        memory = cls(file_path=path, max_lessons=max_lessons)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            if isinstance(data, dict):
                raw_lessons = data.get("lessons")
                if isinstance(raw_lessons, list):
                    memory.lessons = [str(item).strip() for item in raw_lessons if str(item).strip()]
                summary = data.get("rolling_summary")
                if isinstance(summary, str):
                    memory.rolling_summary = summary.strip()
                updated = data.get("updated")
                if isinstance(updated, (int, float)):
                    memory.updated = float(updated)
        return memory

    def save(self) -> None:
        self.updated = time.time()
        payload = {
            "lessons": self.lessons,
            "rolling_summary": self.rolling_summary,
            "updated": self.updated,
        }
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # Memory is best-effort; never crash the agent over a failed write.
            pass

    def add_lesson(self, text: str) -> bool:
        """Append a durable lesson. Returns False if empty or a duplicate."""
        note = (text or "").strip()
        if not note:
            return False
        if note in self.lessons:
            return False
        self.lessons.append(note)
        if len(self.lessons) > self.max_lessons:
            # Keep the most recent lessons when over the cap.
            self.lessons = self.lessons[-self.max_lessons:]
        self.save()
        return True

    def set_summary(self, text: str) -> None:
        self.rolling_summary = (text or "").strip()
        self.save()

    def clear(self) -> None:
        self.lessons = []
        self.rolling_summary = ""
        self.save()

    def is_empty(self) -> bool:
        return not self.lessons and not self.rolling_summary

    def render_for_prompt(self, max_chars: int = 8000) -> str:
        """Render memory as a system-prompt section, or empty string if none."""
        if self.is_empty():
            return ""
        parts: List[str] = ["## Persistent memory (carried across sessions)"]
        if self.rolling_summary:
            parts.append(
                "Summary of earlier conversation that scrolled out of context:\n"
                + self.rolling_summary
            )
        if self.lessons:
            bullets = "\n".join(f"- {lesson}" for lesson in self.lessons)
            parts.append("Lessons and facts to remember and apply:\n" + bullets)
        block = "\n\n".join(parts)
        if len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n...[memory truncated]..."
        return block
