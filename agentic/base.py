"""Shared plumbing for the `agentic` package.

Every technique module in this package talks to the model through the same
duck-typed interface used elsewhere in the project: an object with

    chat(messages: list[dict], tools: list[dict] | None = None) -> dict

returning at least a ``"content"`` key. Tests use a scripted stub with the
same shape, so nothing here imports ``models.py`` (and nothing here needs a
network connection to be testable).

Also holds the small text utilities that every strategy needs: robust JSON
extraction from chatty model replies, 0-10 score parsing, final-answer
extraction, majority voting for self-consistency, and a token estimate
matching the rest of the project (bytes/4).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


class TextModel(Protocol):
    """Minimal model interface every technique depends on."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]: ...


Message = Dict[str, Any]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def complete(model: Any, messages: List[Message]) -> str:
    """One plain chat call, returning stripped text content.

    Accepts either a model object (``.chat``) or a bare callable taking
    messages and returning text, so tests and alternative backends are cheap.
    """
    if hasattr(model, "chat"):
        response = model.chat(messages)
    elif callable(model):
        response = model(messages)
    else:  # pragma: no cover - defensive
        raise TypeError("model must expose chat(messages) or be callable")
    if isinstance(response, Mapping):
        return str(response.get("content") or "").strip()
    return str(response or "").strip()


def ask(model: Any, system: str, user: str) -> str:
    """Single system+user call."""
    return complete(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


def estimate_tokens(text: str) -> int:
    """Same rough accounting the rest of the project uses (chars/4)."""
    return max(1, len(text or "") // 4)


def truncate(text: str, limit: int) -> str:
    """Clip text to ``limit`` chars with an explicit marker."""
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + " …[truncated]"


def stable_hash(text: str) -> str:
    """Deterministic short id (Python's hash() is salted per process)."""
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_json(text: str) -> Optional[Any]:
    """Pull the first decodable JSON value out of a model reply.

    Handles fenced blocks, bare objects/arrays, and trailing prose. Returns
    None when nothing decodes; callers decide the fallback.
    """
    if not text:
        return None
    candidates: List[str] = [m.group(1).strip() for m in _FENCE.finditer(text)]
    candidates.append(text.strip())
    # Balanced scan for object/array literals inside the raw text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            closing = -1
            for index in range(start, len(text)):
                char = text[index]
                in_string, escaped = _scan_char(char, in_string, escaped)
                if in_string:
                    continue
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        closing = index
                        break
            if closing == -1:
                break
            candidates.append(text[start : closing + 1])
            start = text.find(opener, closing + 1)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _scan_char(char: str, in_string: bool, escaped: bool) -> Tuple[bool, bool]:
    """Advance the JSON string-tracking state machine by one character."""
    if in_string:
        if escaped:
            return True, False
        if char == "\\":
            return True, True
        if char == '"':
            return False, False
        return True, False
    if char == '"':
        return True, False
    return False, False


def parse_score(text: str, default: float = 0.5) -> float:
    """Read a 0-1 or 0-10 score out of a reply and normalise to 0-1.

    A reply like "8/10" or "Rating: 8" becomes 0.8. Values already in 0-1
    stay put. Unparseable replies get ``default``.
    """
    if not text:
        return default
    match = _NUMBER.search(text)
    if not match:
        return default
    try:
        value = float(match.group())
    except ValueError:
        return default
    if 0.0 <= value <= 1.0 and re.search(r"\b0?\.\d|^1(\.0+)?$", match.group()):
        return value
    return max(0.0, min(1.0, value / 10.0))


_FINAL_MARKERS = (
    "FINAL ANSWER:",
    "FINAL:",
    "Answer:",
    "ANSWER:",
    "Final answer:",
)


def extract_final(text: str, fallback: Optional[str] = None) -> str:
    """Find the deliverable inside a reasoning trace.

    Prefers an explicit FINAL/Answer marker (the last one wins, since models
    often restate). Otherwise returns the last non-empty paragraph. Falls back
    to the whole text, then ``fallback``.
    """
    text = (text or "").strip()
    if not text:
        return fallback or ""
    best = -1
    best_end = -1
    for marker in _FINAL_MARKERS:
        index = text.rfind(marker)
        if index > best:
            best = index
            best_end = index + len(marker)
    if best != -1:
        tail = text[best_end:].strip()
        if tail:
            return tail.strip().strip("`").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs[-1] if paragraphs else (fallback or text)


def normalize_key(text: str) -> str:
    """Loose normalisation used for majority voting over answer variants."""
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    text = re.sub(r"[^a-z0-9一-鿿 ]+", "", text)
    return text[:400]


def majority_vote(candidates: Sequence[str]) -> Tuple[str, int]:
    """Most common normalised answer; returns (answer, votes).

    Ties break toward the earliest candidate so behaviour stays deterministic.
    """
    if not candidates:
        return "", 0
    counts: Dict[str, int] = {}
    first_text: Dict[str, str] = {}
    order: List[str] = []
    for candidate in candidates:
        key = normalize_key(candidate)
        if key not in counts:
            counts[key] = 0
            order.append(key)
            first_text[key] = candidate
        counts[key] += 1
    best_key = max(order, key=lambda k: counts[k])
    return first_text[best_key], counts[best_key]


def dedupe(items: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication on stripped strings."""
    seen = set()
    out: List[str] = []
    for item in items:
        value = (item or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for zero vectors or mismatched lengths."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


# ---------------------------------------------------------------------------
# Prompt fragments reused by several strategies
# ---------------------------------------------------------------------------

CRITIC_RUBRIC = (
    "Score the work against concrete criteria: correctness, completeness, "
    "and whether it actually satisfies the task. Ignore style preferences."
)


def judge_prompt(task: str, candidate: str) -> str:
    return (
        f"TASK:\n{task}\n\nCANDIDATE:\n{candidate}\n\n"
        f"{CRITIC_RUBRIC}\n"
        "Reply with a single line: SCORE: <0-10>, then one sentence of reason."
    )


def bullet_list(items: Sequence[str], limit: int = 12) -> str:
    """Render up to ``limit`` items as '- ' bullets."""
    return "\n".join(f"- {item}" for item in list(items)[:limit])


@dataclass
class TraceEvent:
    """One step of a strategy, kept for debugging and for the UI."""

    step: int
    kind: str
    detail: str
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def render_trace(events: Sequence[TraceEvent], max_events: int = 30) -> str:
    lines = []
    for event in list(events)[-max_events:]:
        score = f" score={event.score:.2f}" if event.score is not None else ""
        lines.append(f"[{event.step}] {event.kind}{score}: {truncate(event.detail, 200)}")
    return "\n".join(lines)
