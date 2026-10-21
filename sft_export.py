"""Export this agent's own saved sessions as SFT (supervised fine-tuning)
training data.

Honest scope: this is NOT model training. This project calls hosted or
local models over an API; it has no GPU training loop and doesn't ship one
here. What this does is turn real, already-had conversations into the
standard chat-SFT JSONL shape ({"messages": [...]} per line, one line per
conversation) that an actual training pipeline - Ollama's own fine-tuning,
an OpenAI-compatible fine-tuning endpoint, axolotl, etc. - can consume
directly. The "pre-training" half of the request has no buildable
counterpart inside an API-calling agent; this is the real, usable piece of
it.

Only plain user/assistant text turns are kept. Tool calls and tool-result
messages are dropped rather than reproduced: faithfully exporting this
project's own JSON-in-text tool-call convention (and native tool_calls
where backends use it) as training examples would teach a fine-tuned model
this project's specific tool-calling shape, not general chat ability, and
mixing the two formats inconsistently across sessions would make for a
worse dataset than leaving tool mechanics out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Replies that are placeholders, not real answers worth training on.
_SKIP_PREFIXES = (
    "(interrupted",
    "(no response",
    "(agent reached the maximum",
    "(自主模式没有返回内容)",
    "(已停止)",
    "sorry, i encountered an error",
    "抱歉，发生了错误",
)


def _is_trainable_reply(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    text = content.strip()
    if len(text) < 2:
        return False
    return not text.lower().startswith(_SKIP_PREFIXES)


def extract_pairs(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Pull clean (system?, user, assistant) turns out of one session's raw
    history, in order. Tool-calling turns are skipped, not merged in."""
    system = next(
        (m.get("content") for m in history if m.get("role") == "system" and m.get("content")),
        None,
    )
    messages: List[Dict[str, str]] = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    pending_user: Optional[str] = None
    for entry in history:
        role = entry.get("role")
        content = entry.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            pending_user = content.strip()
        elif role == "assistant" and "tool_calls" not in entry:
            if pending_user is not None and _is_trainable_reply(content):
                messages.append({"role": "user", "content": pending_user})
                messages.append({"role": "assistant", "content": content.strip()})
            pending_user = None
        elif role == "tool":
            # A tool round-trip orphaned pending_user (the reply that will
            # eventually answer it isn't a clean single-turn exchange
            # anymore); drop it rather than pair it with the wrong reply.
            pending_user = None
    return messages


def iter_session_files(sessions_dir: Path) -> Iterable[Path]:
    if not sessions_dir.is_dir():
        return []
    return sorted(sessions_dir.glob("*.json"))


def export_sft_dataset(sessions_dir: Path, out_path: Path, min_pairs: int = 1) -> Dict[str, int]:
    """Write one JSONL line per session that has at least `min_pairs` clean
    user/assistant exchanges. Returns counts for reporting, not just a total,
    so a caller can tell "nothing to export" apart from "export failed"."""
    sessions_dir = Path(sessions_dir)
    out_path = Path(out_path)
    stats = {"sessions_seen": 0, "sessions_exported": 0, "pairs_exported": 0}

    lines: List[str] = []
    for path in iter_session_files(sessions_dir):
        stats["sessions_seen"] += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = data.get("history")
        if not isinstance(history, list):
            continue
        messages = extract_pairs(history)
        pair_count = sum(1 for m in messages if m["role"] == "assistant")
        if pair_count < max(1, min_pairs):
            continue
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
        stats["sessions_exported"] += 1
        stats["pairs_exported"] += pair_count

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return stats
