"""Extended tool capabilities: code execution, web retrieval, GUI control.

Three tools the base agent does not have:

  CodeInterpreter   a persistent Python worker subprocess, so state (imports,
                    variables, loaded dataframes) survives between snippets
                    like a notebook kernel. Falls back to a one-shot run when
                    the worker dies.
  WebBrowser        search -> fetch -> chunk -> rank -> cite. This is the
                    "web browsing / RAG search" pair: retrieval over live
                    pages, with the URL attached to every passage so the model
                    can cite and you can check.
  ComputerUse       screen capture plus mouse/keyboard control through
                    pyautogui when it is installed. Every mutating action is
                    gated behind an explicit confirmation callback, because
                    driving a real desktop is both powerful and irreversible.

All three degrade honestly: missing optional dependency or blocked action
returns a plain reason rather than raising into the agent loop.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic.base import cosine, truncate


# ---------------------------------------------------------------------------
# Code interpreter
# ---------------------------------------------------------------------------

WORKER_SOURCE = r'''
import json, sys, traceback, io, contextlib

scope = {"__name__": "__main__"}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if request.get("op") == "exit":
        print(json.dumps({"ok": True}), flush=True)
        break
    code = request.get("code", "")
    buffer = io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(code, "<snippet>", "exec"), scope)
    except Exception:
        ok = False
        buffer.write(traceback.format_exc())
    output = buffer.getvalue()
    limit = int(request.get("limit") or 20000)
    if len(output) > limit:
        output = output[:limit] + "\n...[output truncated]"
    result = None
    for name in ("result", "answer", "output"):
        if name in scope and not name.startswith("__"):
            try:
                result = repr(scope[name])
            except Exception:
                result = "<unrepresentable>"
            break
    print(json.dumps({"ok": ok, "output": output, "result": result}), flush=True)
'''


@dataclass
class CodeResult:
    ok: bool
    output: str
    result: Optional[str] = None
    duration: float = 0.0
    mode: str = "persistent"


class CodeInterpreter:
    """Persistent Python REPL driven over stdin/stdout JSON lines.

    A long-lived worker keeps variables between calls, which matters for data
    work (load once, query many times). It is not a security sandbox: code runs
    with the agent's own OS permissions. Use it for analysis, not for untrusted
    input. A worker that dies restarts on the next call; a worker that hangs is
    killed by ``close()``.
    """

    def __init__(
        self,
        python: Optional[str] = None,
        timeout: float = 30.0,
        cwd: Optional[Path] = None,
        output_limit: int = 20000,
    ) -> None:
        self.python = python or sys.executable
        self.timeout = timeout
        self.cwd = str(cwd) if cwd else None
        self.output_limit = output_limit
        self._proc: Optional[subprocess.Popen] = None
        self._script: Optional[Path] = None

    # -- lifecycle ---------------------------------------------------------

    def _ensure_worker(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        try:
            if self._script is None:
                handle = tempfile.NamedTemporaryFile(
                    "w", suffix="_agentic_worker.py", delete=False, encoding="utf-8"
                )
                handle.write(WORKER_SOURCE)
                handle.close()
                self._script = Path(handle.name)
            self._proc = subprocess.Popen(
                [self.python, "-u", str(self._script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                cwd=self.cwd,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return True
        except OSError:
            self._proc = None
            return False

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(json.dumps({"op": "exit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        if self._script and self._script.exists():
            try:
                self._script.unlink()
            except OSError:
                pass
            self._script = None

    def __enter__(self) -> "CodeInterpreter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- execution ---------------------------------------------------------

    def run(self, code: str) -> CodeResult:
        """Execute a snippet, preferring the persistent worker."""
        started = time.time()
        if self._ensure_worker():
            result = self._run_persistent(code, started)
            if result is not None:
                return result
        return self._run_oneshot(code, started)

    def _run_persistent(self, code: str, started: float) -> Optional[CodeResult]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        try:
            proc.stdin.write(json.dumps({"code": code, "limit": self.output_limit}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.close()
            return None
        try:
            line = proc.stdout.readline()
        except Exception:
            self.close()
            return None
        if not line:
            self.close()
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.close()
            return None
        return CodeResult(
            ok=bool(payload.get("ok")),
            output=str(payload.get("output") or ""),
            result=payload.get("result"),
            duration=time.time() - started,
            mode="persistent",
        )

    def _run_oneshot(self, code: str, started: float) -> CodeResult:
        try:
            completed = subprocess.run(
                [self.python, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                cwd=self.cwd,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return CodeResult(False, f"Timed out after {self.timeout:.0f}s.", None,
                              time.time() - started, "oneshot")
        except OSError as exc:
            return CodeResult(False, f"Cannot start Python: {exc}", None,
                              time.time() - started, "oneshot")
        output = (completed.stdout or "") + (
            ("\n" + completed.stderr) if completed.stderr else ""
        )
        return CodeResult(
            ok=completed.returncode == 0,
            output=truncate(output.strip(), self.output_limit),
            result=None,
            duration=time.time() - started,
            mode="oneshot",
        )


def format_code_result(result: CodeResult) -> str:
    """Render a CodeResult for a tool result block."""
    parts = [f"[{result.mode}] {'ok' if result.ok else 'error'} in {result.duration:.2f}s"]
    if result.output:
        parts.append(result.output)
    if result.result:
        parts.append(f"result: {result.result}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Web browsing / retrieval
# ---------------------------------------------------------------------------

CHUNK_CHARS = 1200


@dataclass
class Passage:
    url: str
    title: str
    text: str
    score: float = 0.0


class WebBrowser:
    """Search, fetch, chunk, rank, cite.

    ``search_fn`` and ``fetch_fn`` default to this project's existing
    ``tools.search_web`` / ``tools.fetch_url``; inject fakes in tests. Ranking
    uses the same hashed embedding as memory, so a passage is scored by lexical
    overlap with the query — cheap, offline, and good enough to pick the right
    paragraph out of a page.
    """

    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], str]] = None,
        fetch_fn: Optional[Callable[[str], str]] = None,
        max_pages: int = 3,
        chunk_chars: int = CHUNK_CHARS,
    ) -> None:
        self.search_fn = search_fn or _default_search
        self.fetch_fn = fetch_fn or _default_fetch
        self.max_pages = max(1, max_pages)
        self.chunk_chars = max(200, chunk_chars)

    def search(self, query: str, max_results: int = 5) -> List[Tuple[str, str]]:
        """Return [(url, title)] from the search backend."""
        try:
            raw = self.search_fn(query, max_results)
        except Exception as exc:
            return [("", f"search failed: {exc}")]
        titled: List[Tuple[str, str]] = []
        lines = (raw or "").splitlines()
        for index, line in enumerate(lines):
            match = re.search(r"URL:\s*(\S+)", line)
            if not match:
                continue
            title = ""
            if index > 0:
                title_match = re.match(r"\s*\d+\.\s+(.*)", lines[index - 1])
                if title_match:
                    title = title_match.group(1).strip()
            titled.append((match.group(1).strip(), title))
        return titled[: max(1, max_results)]

    def fetch_passages(self, url: str, query: str = "") -> List[Passage]:
        """Fetch one URL and return its chunks ranked against the query."""
        try:
            text = self.fetch_fn(url)
        except Exception as exc:
            return [Passage(url, "fetch failed", f"Could not fetch: {exc}", 0.0)]
        return self.rank_chunks(url, text, query)

    def rank_chunks(self, url: str, text: str, query: str, top: int = 4) -> List[Passage]:
        """Split text into passages and rank them by embedding similarity."""
        from agentic.memory_store import hashed_embed

        text = text or ""
        chunks: List[str] = []
        buffer: List[str] = []
        length = 0
        for line in text.splitlines(keepends=True):
            if buffer and length + len(line) > self.chunk_chars:
                chunks.append("".join(buffer).strip())
                buffer, length = [], 0
            buffer.append(line)
            length += len(line)
        if buffer:
            chunks.append("".join(buffer).strip())
        passages = [Passage(url, "", chunk) for chunk in chunks if len(chunk) > 80]
        if not passages:
            return [Passage(url, "", truncate(text, self.chunk_chars), 0.0)]
        if query:
            qvec = hashed_embed(query)
            for passage in passages:
                passage.score = cosine(qvec, hashed_embed(passage.text))
            passages.sort(key=lambda p: p.score, reverse=True)
        return passages[: max(1, top)]

    def research(self, query: str, max_results: int = 3, top_per_page: int = 2) -> str:
        """Search, fetch the top pages, and return the best cited passages.

        This is the full browse-and-retrieve path a model should call instead
        of fetching a URL blindly: every returned passage carries its URL.
        """
        results = self.search(query, max_results)
        if not results:
            return "No search results."
        blocks: List[str] = []
        for url, title in results[: self.max_pages]:
            if not url:
                blocks.append(title)
                continue
            passages = self.fetch_passages(url, query)[:top_per_page]
            if not passages:
                continue
            header = f"### {title or url}\n{url}"
            body = "\n\n".join(
                f"[{index + 1}] (score {p.score:.2f}) {truncate(p.text, 1500)}"
                for index, p in enumerate(passages)
            )
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks) if blocks else "Fetched pages had no usable text."


def _default_search(query: str, max_results: int) -> str:
    try:
        from tools import search_web

        return search_web(query, max_results=max_results)
    except Exception as exc:
        return f"Search unavailable: {exc}"


def _default_fetch(url: str) -> str:
    try:
        from tools import fetch_url

        return fetch_url(url)
    except Exception as exc:
        return f"Fetch failed: {exc}"


# ---------------------------------------------------------------------------
# Computer use / GUI automation
# ---------------------------------------------------------------------------

@dataclass
class GuiAction:
    """One desktop action. ``mutating`` actions need confirmation."""

    kind: str
    args: Dict[str, Any] = field(default_factory=dict)
    mutating: bool = True


MUTATING_KINDS = {
    "click", "double_click", "right_click", "type", "hotkey", "scroll", "drag", "press",
}


class ComputerUse:
    """Screen + mouse + keyboard control, guarded by a confirmation callback.

    ``confirm(action) -> bool`` must return True for any mutating action; pass
    None and this module refuses to touch the desktop at all. That default is
    deliberate: an unattended agent that can click arbitrary coordinates can
    delete files or send messages, and there is no undo. Read-only actions
    (screenshot, screen size) are always allowed.

    Requires ``pyautogui`` for control; screen size falls back to ctypes on
    Windows so the agent can still report geometry without the dependency.
    """

    def __init__(self, confirm: Optional[Callable[[GuiAction], bool]] = None) -> None:
        self.confirm = confirm

    @staticmethod
    def available() -> bool:
        try:
            import pyautogui  # noqa: F401

            return True
        except Exception:
            return False

    def screen_size(self) -> Tuple[int, int]:
        try:
            import pyautogui

            size = pyautogui.size()
            return int(size.width), int(size.height)
        except Exception:
            pass
        try:  # Windows fallback without pyautogui
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return (0, 0)

    def screenshot(self, path: Optional[str] = None) -> str:
        """Capture the screen. Read-only, so no confirmation needed."""
        try:
            import pyautogui
        except Exception:
            return "pyautogui is not installed; install it to capture the screen."
        target = path or str(
            Path(tempfile.gettempdir()) / f"agentic_shot_{int(time.time())}.png"
        )
        try:
            pyautogui.screenshot(target)
            return f"Screenshot saved to {target}"
        except Exception as exc:
            return f"Screenshot failed: {exc}"

    def _gate(self, action: GuiAction) -> Optional[str]:
        if action.mutating and self.confirm is None:
            return (
                f"Refused: '{action.kind}' changes the desktop and no confirmation "
                "callback is configured."
            )
        if action.mutating and self.confirm is not None and not self.confirm(action):
            return f"Refused by user: '{action.kind}' was not performed."
        return None

    def perform(self, action: GuiAction) -> str:
        """Execute one action after the gate; returns a human-readable result."""
        blocked = self._gate(action)
        if blocked:
            return blocked
        try:
            import pyautogui
        except Exception:
            return "pyautogui is not installed; GUI automation is unavailable."
        try:
            kind = action.kind
            args = action.args
            if kind == "click":
                pyautogui.click(
                    int(args.get("x", 0)), int(args.get("y", 0)),
                    button=args.get("button", "left"),
                )
            elif kind == "double_click":
                pyautogui.doubleClick(int(args.get("x", 0)), int(args.get("y", 0)))
            elif kind == "right_click":
                pyautogui.rightClick(int(args.get("x", 0)), int(args.get("y", 0)))
            elif kind == "type":
                pyautogui.typewrite(
                    str(args.get("text", "")), interval=float(args.get("interval", 0.02))
                )
            elif kind == "press":
                pyautogui.press(str(args.get("key", "enter")))
            elif kind == "hotkey":
                keys = args.get("keys") or []
                pyautogui.hotkey(*[str(k) for k in keys])
            elif kind == "scroll":
                pyautogui.scroll(int(args.get("clicks", 0)))
            elif kind == "drag":
                pyautogui.moveTo(int(args.get("x1", 0)), int(args.get("y1", 0)))
                pyautogui.dragTo(
                    int(args.get("x2", 0)), int(args.get("y2", 0)),
                    duration=float(args.get("duration", 0.3)),
                )
            elif kind == "move":
                pyautogui.moveTo(int(args.get("x", 0)), int(args.get("y", 0)))
            else:
                return f"Unknown GUI action: {kind}"
            return f"Performed {kind}."
        except Exception as exc:
            return f"GUI action '{action.kind}' failed: {exc}"


COMPUTER_USE_ACTIONS = sorted(MUTATING_KINDS | {"move", "screenshot", "size"})
