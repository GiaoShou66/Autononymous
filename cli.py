from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4
import os

import sys
import time
import difflib
import atexit

try:
    import readline  # Unix only; history + tab completion degrade gracefully without it
except ImportError:
    readline = None

try:
    from pygments import lex
    from pygments.lexers import get_lexer_for_filename, TextLexer
    from pygments.token import Token
except ImportError:
    lex = None
    Token = None
    TextLexer = None
    get_lexer_for_filename = None

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from agent import Agent
from autonomous import AutonomousAgent, AutonomousConfig
from config import load_config
from memory import AgentMemory
from models import ModelClient
from hooks import HookManager, load_settings, settings_path
from sessions import SessionStore
from skills import SkillRegistry
from tools import ProjectTools, ChangeTracker, fetch_url, search_web
from media import analyze_media, transcribe_audio, analyze_video, encode_image_base64, get_image_info
from mcp_client import MCPClientManager
from data_analysis import analyze_dataset


app = typer.Typer(help="Local AI code helper agent")
console = Console()


PROJECT_INSTRUCTION_FILES = ["CLAUDE.md", "AGENTS.md", ".agentrules", "AGENT.md"]
CUSTOM_COMMANDS_DIR = ".commands"


def _load_project_instructions(project_root: Path) -> str:
    """Read CLAUDE.md / AGENTS.md style project rules from the project root."""
    blocks: list[str] = []
    for name in PROJECT_INSTRUCTION_FILES:
        path = project_root / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                blocks.append(f"# From {name}\n{text}")
    return "\n\n".join(blocks)


def _load_custom_commands(project_root: Path) -> dict[str, str]:
    """Load user slash commands from .commands/*.md (name -> prompt template)."""
    commands: dict[str, str] = {}
    cmd_dir = project_root / CUSTOM_COMMANDS_DIR
    if cmd_dir.is_dir():
        for path in sorted(cmd_dir.glob("*.md")):
            try:
                commands[path.stem.lower()] = path.read_text(encoding="utf-8")
            except OSError:
                continue
    return commands


def _expand_file_mentions(text: str, project_root: Path) -> str:
    """Replace @path tokens with the referenced file's contents inline."""
    import re

    def repl(match: "re.Match[str]") -> str:
        rel = match.group(1)
        candidate = (project_root / rel).resolve()
        try:
            candidate.relative_to(project_root.resolve())
        except ValueError:
            return match.group(0)
        if candidate.is_file():
            try:
                body = candidate.read_text(encoding="utf-8")
            except OSError:
                return match.group(0)
            return f"{match.group(0)}\n\n--- {rel} ---\n{body}\n--- end {rel} ---\n"
        return match.group(0)

    return re.sub(r"@([\w./\\-]+)", repl, text)


UPLOAD_MAX_CHARS = 100000
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_MEDIA_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _ingest_upload(path: Path, model_client) -> str:
    """
    Read an uploaded file and return a text block to inject into the chat so
    the agent can see it. Text/code is inlined; images and audio/video are run
    through the media analyzers to produce a text description.
    """
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        try:
            b64, mime = encode_image_base64(str(path))
            desc = model_client.describe_image(
                image_base64=b64, prompt="Describe this image in detail.", mime_type=mime
            )
            meta = get_image_info(str(path))
            return f"[image] {meta}\n\nAI description:\n{desc}"
        except Exception as exc:
            return f"[image] {path.name} (vision analysis failed: {exc})"
    if ext in _MEDIA_EXTS:
        try:
            return f"[media]\n{analyze_media(str(path))}"
        except Exception as exc:
            return f"[media] {path.name} (analysis failed: {exc})"
    # Default: treat as text/code.
    try:
        body = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[binary file] {path.name} ({path.stat().st_size} bytes) — not a text file; cannot inline."
    except OSError as exc:
        return f"[error reading {path.name}: {exc}]"
    truncated = ""
    if len(body) > UPLOAD_MAX_CHARS:
        truncated = f"\n...[truncated, {len(body) - UPLOAD_MAX_CHARS} more chars]"
        body = body[:UPLOAD_MAX_CHARS]
    return f"```\n{body}{truncated}\n```"


def _paste_from_clipboard(model_client) -> tuple[str, str] | None:
    """
    Read the OS clipboard and return (label, text_block) to inject into chat.
    Handles a copied image or copied files (needs Pillow) and plain text
    (via tkinter, no extra deps). Returns None if the clipboard is empty.
    """
    # Image or copied files via Pillow, if available.
    try:
        from PIL import ImageGrab, Image  # type: ignore

        data = ImageGrab.grabclipboard()
        if isinstance(data, list) and data:
            path = Path(data[0])
            if path.is_file():
                return f"clipboard file: {path.name}", _ingest_upload(path, model_client)
        elif isinstance(data, Image.Image):
            import tempfile

            tmp = Path(tempfile.gettempdir()) / "agent_clipboard.png"
            data.save(tmp, "PNG")
            return "clipboard image", _ingest_upload(tmp, model_client)
    except Exception:
        pass  # Pillow missing or clipboard not an image; fall back to text.

    # Plain text via tkinter (always available in CPython).
    try:
        import tkinter as _tk

        root = _tk.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        except _tk.TclError:
            text = ""
        root.destroy()
    except Exception:
        text = ""

    text = (text or "").strip()
    if not text:
        return None
    truncated = ""
    if len(text) > UPLOAD_MAX_CHARS:
        truncated = f"\n...[truncated, {len(text) - UPLOAD_MAX_CHARS} more chars]"
        text = text[:UPLOAD_MAX_CHARS]
    return "clipboard text", f"```\n{text}{truncated}\n```"


def _split_path_and_prompt(rest: str) -> tuple[str, str]:
    """Parse `/upload` args: a quoted path, or first token as path, rest as prompt."""
    rest = rest.strip()
    if rest.startswith('"') and '"' in rest[1:]:
        end = rest.index('"', 1)
        return rest[1:end], rest[end + 1 :].strip()
    parts = rest.split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _print_help() -> None:
    console.print("[bold]Commands[/bold]")
    # markup=False so argument placeholders like [name], [file], [s] are shown
    # literally and not parsed as Rich tags ([s] = strikethrough, which used to
    # strike through the rest of the help).
    console.print(
        "  /exit                 Exit the agent\n"
        "  /help                 Show this help\n"
        "  /http <url>           Fetch a URL over HTTP/HTTPS and show the text response\n"
        "  /search <query>       Search the web using DuckDuckGo\n"
        "  /delete <path>        Delete a file from the project\n"
        "  /mkdir <path>         Create a directory in the project\n"
        "  /changes              Show all file changes made this session\n"
        "  /changes clear        Clear the changes log\n"
        "  /skills [query]       List skills, or search for matching skills\n"
        "  /use <skill>          Keep a skill loaded for future requests\n"
        "  /clear-skills         Clear manually loaded skills\n"
        "  /auto-skills on|off   Enable or disable automatic skill matching\n"
        "  /image(s) <path> [prompt]  Analyze picture with vision model\n"
        "  /audio <path>           Transcribe and analyze audio (MP3/WAV/FLAC)\n"
        "  /data <path> [chart]    Profile CSV/TSV/JSON data and optionally chart it\n"
        "  /video(s) <path> [prompt]  Analyze video (extract key frames + audio transcript)\n"
        "  /media <path>           Auto-detect media type and analyze\n"
        "  /diff [file]          Show git diff for project or file\n"
        "  /git <command>        Run git status, commit, log, branch\n"
        "  /undo [file]          Revert last file modification from .bak\n"
        "  /grep <pattern>       Search codebase for regex/text with line numbers\n"
        "  /outline <file>       Show classes, functions, and symbols in file\n"
        "  /lines <file> [s] [e] Read specific lines with line numbers\n"
        "  /syntax <file>        Validate Python/JSON/YAML syntax\n"
        "  /test [command]       Run test suite (pytest/unittest)\n"
        "  /clear                Reset conversation context (keeps memory/tasks)\n"
        "  /init                 Scan project and generate a CLAUDE.md\n"
        "  /compact              Compress old turns into persistent memory\n"
        "  /memory               Show persistent memory (lessons + rolling summary)\n"
        "  /remember <note>      Save a durable lesson to persistent memory\n"
        "  /reflect              Extract lessons from this session into memory\n"
        "  /forget all|<n>       Clear all memory, or forget lesson number n\n"
        "  /plan on|off          Plan mode: agent plans, makes no changes\n"
        "  /approve on|off       Approve each file change before it applies\n"
        "  /hooks [reload]       Show lifecycle hooks from .agent/settings.json\n"
        "  !<command>            Run a shell command directly\n"
        "  #<note>               Save a note to persistent memory\n"
        "  /todo                 Show the agent's task list\n"
        "  /task add|done|clear  Manage tasks (/task add <text>, /task done <n>)\n"
        "  /save [name]          Save this conversation as a session\n"
        "  /go [# or id]         Pick a session to switch into (saves the current one)\n"
        "  /resume <id>          Resume a saved session\n"
        "  /sessions             List saved sessions\n"
        "  /new [title]          Start a fresh conversation\n"
        "  /rename <title>       Rename the current conversation\n"
        "  /delete-session <id>  Delete a saved conversation\n"
        "  /tokens               Show estimated token usage this session\n"
        "  /model [name]         Show or switch the active model\n"
        "  /reasoning on|off     Toggle native local-model reasoning\n"
        "  /routing              Show fast/reasoning model routing\n"
        "  /status               Show session status dashboard\n"
        "  /review [file]        Review the current git diff\n"
        "  /commit [message]     Stage all and commit (auto-message if omitted)\n"
        "  /typing on|off [cps]  Toggle the typewriter animation / set speed\n"
        "  /doctor               Check Ollama connectivity and model install\n"
        "  /export [path]        Export the conversation to a Markdown file\n"
        "  /export-sft [path]    Export saved sessions as SFT training data (JSONL)\n"
        "  /agent <task>         Delegate a subtask to a fresh subagent\n"
        "  /loop [cycles=N] [every=S] <goal>  Run a goal autonomously until done\n"
        "  /refine [rounds=N] <prompt>  Answer, then critique and revise (default 1 round)\n"
        "  /best [n=N] <prompt>  Sample N answers and keep the best (default 3, text only)\n"
        "  @path                Inline a file's contents into your message\n"
        "  /upload <path> [question]  Upload a file (text/image/audio/video) into the chat\n"
        "  /paste [question]     Paste clipboard (text, or image/files with Pillow) into the chat\n"
        "  /modes                List available agent modes (claude-code, codex, etc.)\n"
        "  /mode <name>          Switch mode: claude-code, codex, coder, reviewer...\n"
        "  /models               List models exposed by the current provider\n"
        "  /limits               Show current capacity settings\n"
        "  /mcp                  Show configured MCP servers and discovered tools\n"
        "\n"
        "[bold cyan]Autonomous Features:[/bold cyan]\n"
        "  /continuous <task>    Continuous coding with auto-continuation\n"
        "  /iterate <task>       Iterative generation with self-improvement\n"
        "  /autonomous <task>    Full autonomous mode with task decomposition\n"
        "  /stream <task>        Long-context streaming generation\n"
        "  /autocorrect <file>   Auto-correct code errors\n"
        "\n"
        "Anything else is sent to the AI assistant, which can automatically\n"
        "load relevant skills, read/write/create/delete files, run commands,\n"
        "and debug your code. Use /changes to review what the AI modified.",
        markup=False,
        highlight=False,
    )


DIFF_TOKEN_COLORS = {
    Token.Keyword: "#ff7b72" if Token else "red",
    Token.Keyword.Constant: "#79c0ff" if Token else "cyan",
    Token.Name.Function: "#d2a8ff" if Token else "magenta",
    Token.Name.Class: "#ffa657" if Token else "yellow",
    Token.Name.Builtin: "#79c0ff" if Token else "cyan",
    Token.String: "#a5d6ff" if Token else "green",
    Token.Comment: "#8b949e" if Token else "dim",
    Token.Number: "#79c0ff" if Token else "cyan",
    Token.Operator: "#ff7b72" if Token else "white",
    Token.Punctuation: "#e6edf3" if Token else "white",
} if Token else {}


def _render_code_diff(
    action: str,
    display_path: str,
    pre_text: str = "",
    post_text: str = "",
    old_snippet: Optional[str] = None,
    new_snippet: Optional[str] = None,
    old_start_line: int = 1,
) -> None:
    """Render a Claude Code / Codex style surgical code diff in the terminal."""
    lexer = None
    if get_lexer_for_filename:
        try:
            lexer = get_lexer_for_filename(display_path)
        except Exception:
            if TextLexer:
                lexer = TextLexer()

    term_width = max(80, min(console.width or 95, 120))
    pre_lines = pre_text.splitlines()
    post_lines = post_text.splitlines()

    added_count = 0
    removed_count = 0
    all_rows = []

    if old_snippet is not None and new_snippet is not None:
        # Patch diff with surrounding context lines
        if not pre_text and old_snippet:
            ctx_before_lines = []
            ctx_before_start = old_start_line
        else:
            ctx_before_lines = pre_lines[max(0, old_start_line - 4):old_start_line - 1]
            ctx_before_start = old_start_line - len(ctx_before_lines)

        old_lines = old_snippet.splitlines()
        new_lines = new_snippet.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines)

        diff_rows = []
        old_line_idx = old_start_line
        new_line_idx = old_start_line

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for line in new_lines[j1:j2]:
                    diff_rows.append((new_line_idx, " ", line, "ctx"))
                    old_line_idx += 1
                    new_line_idx += 1
            elif tag == "delete":
                for line in old_lines[i1:i2]:
                    diff_rows.append((old_line_idx, "-", line, "del"))
                    old_line_idx += 1
                    removed_count += 1
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    diff_rows.append((new_line_idx, "+", line, "add"))
                    new_line_idx += 1
                    added_count += 1
            elif tag == "replace":
                for line in old_lines[i1:i2]:
                    diff_rows.append((old_line_idx, "-", line, "del"))
                    old_line_idx += 1
                    removed_count += 1
                for line in new_lines[j1:j2]:
                    diff_rows.append((new_line_idx, "+", line, "add"))
                    new_line_idx += 1
                    added_count += 1

        ctx_after_start = new_line_idx
        if post_lines:
            ctx_after_lines = post_lines[ctx_after_start - 1:ctx_after_start + 2]
        else:
            ctx_after_lines = []

        for i, line in enumerate(ctx_before_lines):
            all_rows.append((ctx_before_start + i, " ", line, "ctx"))
        all_rows.extend(diff_rows)
        for i, line in enumerate(ctx_after_lines):
            all_rows.append((ctx_after_start + i, " ", line, "ctx"))

    elif not pre_text and post_text:
        # Brand new file created
        added_count = len(post_lines)
        if len(post_lines) <= 45:
            for idx, line in enumerate(post_lines, 1):
                all_rows.append((idx, "+", line, "add"))
        else:
            for idx in range(1, 31):
                all_rows.append((idx, "+", post_lines[idx - 1], "add"))
            all_rows.append((None, "...", f"... ({len(post_lines) - 35} more lines)", "skip"))
            for idx in range(len(post_lines) - 4, len(post_lines) + 1):
                all_rows.append((idx, "+", post_lines[idx - 1], "add"))
    else:
        # Full file replacement / update
        sm = difflib.SequenceMatcher(None, pre_lines, post_lines)
        old_idx = 1
        new_idx = 1
        hunks = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                old_idx += (i2 - i1)
                new_idx += (j2 - j1)
            else:
                hunk_del = [(old_idx + k, "-", pre_lines[i1 + k], "del") for k in range(i2 - i1)]
                hunk_add = [(new_idx + k, "+", post_lines[j1 + k], "add") for k in range(j2 - j1)]
                removed_count += len(hunk_del)
                added_count += len(hunk_add)

                b_start = max(0, i1 - 3)
                ctx_b = [(new_idx - (i1 - b_start) + k, " ", pre_lines[b_start + k], "ctx") for k in range(i1 - b_start)]
                ctx_a = [(new_idx + len(hunk_add) + k, " ", post_lines[j2 + k], "ctx") for k in range(min(3, len(post_lines) - j2))]

                hunks.append((ctx_b, hunk_del, hunk_add, ctx_a))
                old_idx += (i2 - i1)
                new_idx += (j2 - j1)

        for idx, (cb, hd, ha, ca) in enumerate(hunks):
            if idx > 0:
                all_rows.append((None, "...", "...", "skip"))
            all_rows.extend(cb)
            all_rows.extend(hd)
            all_rows.extend(ha)
            all_rows.extend(ca)
            if len(all_rows) > 60:
                all_rows.append((None, "...", "... (additional changes omitted)", "skip"))
                break

    # Format summary header: Added X lines, removed Y lines
    added_str = f"Added {added_count} line" if added_count == 1 else f"Added {added_count} lines"
    removed_str = f"removed {removed_count} line" if removed_count == 1 else f"removed {removed_count} lines"
    if added_count > 0 and removed_count > 0:
        summary = f"{added_str}, {removed_str}"
    elif added_count > 0:
        summary = added_str
    elif removed_count > 0:
        summary = f"Removed {removed_count} line" if removed_count == 1 else f"Removed {removed_count} lines"
    else:
        summary = "No lines changed"

    console.print(f"[bold #10b981]●[/bold #10b981] [bold]{action}[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")
    console.print(f"  [dim]└ {summary}[/dim]")

    max_digits = max(4, len(str(max((r[0] for r in all_rows if r[0] is not None), default=999))))

    for line_no, marker, code, style_type in all_rows:
        if style_type == "skip":
            console.print(f"       [dim]{code}[/dim]")
            continue

        t = Text()
        if style_type == "add":
            bg = "#0c3917"
            prefix = f"  {line_no:>{max_digits}} +  "
            t.append(prefix, style=f"bold #7ee787 on {bg}")
        elif style_type == "del":
            bg = "#4d1417"
            prefix = f"  {line_no:>{max_digits}} -  "
            t.append(prefix, style=f"bold #ffa198 on {bg}")
        else:
            bg = None
            prefix = f"  {line_no:>{max_digits}}    "
            t.append(prefix, style="dim")

        if lex and Token and lexer:
            try:
                tokens = list(lex(code, lexer))
            except Exception:
                tokens = [(Token.Text, code)]

            line_len = len(prefix)
            for tok_type, tok_val in tokens:
                if tok_val == "\n":
                    continue
                color = None
                curr = tok_type
                while curr:
                    if curr in DIFF_TOKEN_COLORS:
                        color = DIFF_TOKEN_COLORS[curr]
                        break
                    curr = curr.parent
                if not color:
                    color = "#e6edf3"
                style = f"{color} on {bg}" if bg else color
                t.append(tok_val, style=style)
                line_len += len(tok_val)
        else:
            style = f"#e6edf3 on {bg}" if bg else "#e6edf3"
            t.append(code, style=style)
            line_len = len(prefix) + len(code)

        if bg and line_len < term_width:
            t.append(" " * (term_width - line_len), style=f"on {bg}")

        console.print(t, soft_wrap=True)


def _on_tool_call(name: str, args: dict, result: str) -> None:
    """Callback to display tool usage in the terminal with Claude Code aesthetic."""
    path_arg = args.get("path") or args.get("file") or ""
    display_path = str(path_arg)
    if ProjectTools.active_instance and path_arg:
        try:
            resolved = ProjectTools.active_instance._resolve_path(path_arg)
            display_path = str(resolved.relative_to(ProjectTools.active_instance.project_root))
        except Exception:
            display_path = str(path_arg)

    if name == "apply_patch":
        if "not found" in result.lower() or "error" in result.lower():
            console.print(f"[bold #10b981]●[/bold #10b981] [bold]Update[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")
            console.print(f"  [red]Failed to apply patch: {escape(result)}[/red]")
            return

        entry = None
        if ProjectTools.active_instance and ProjectTools.active_instance.changes.entries:
            for e in reversed(ProjectTools.active_instance.changes.entries):
                if e.action == "patch" and (e.path == path_arg or path_arg in e.path):
                    entry = e
                    break

        if entry:
            _render_code_diff(
                action="Update",
                display_path=display_path,
                pre_text=entry.pre_text,
                post_text=entry.post_text,
                old_snippet=entry.old_snippet,
                new_snippet=entry.new_snippet,
                old_start_line=entry.old_start_line,
            )
        else:
            _render_code_diff(
                action="Update",
                display_path=display_path,
                old_snippet=args.get("old_snippet", ""),
                new_snippet=args.get("new_snippet", ""),
            )

    elif name == "write_file":
        if "error" in result.lower():
            console.print(f"[bold #10b981]●[/bold #10b981] [bold]Write[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")
            console.print(f"  [red]Error: {escape(result)}[/red]")
            return

        entry = None
        if ProjectTools.active_instance and ProjectTools.active_instance.changes.entries:
            for e in reversed(ProjectTools.active_instance.changes.entries):
                if e.action in {"write", "create"} and (e.path == path_arg or path_arg in e.path):
                    entry = e
                    break

        if entry and entry.action == "create":
            _render_code_diff(action="Create", display_path=display_path, pre_text="", post_text=entry.post_text)
        elif entry:
            _render_code_diff(action="Update", display_path=display_path, pre_text=entry.pre_text, post_text=entry.post_text)
        else:
            content = args.get("content", "")
            _render_code_diff(action="Update", display_path=display_path, pre_text="", post_text=content)

    elif name == "delete_code":
        entry = None
        if ProjectTools.active_instance and ProjectTools.active_instance.changes.entries:
            for e in reversed(ProjectTools.active_instance.changes.entries):
                if e.action == "delete_code" and (e.path == path_arg or path_arg in e.path):
                    entry = e
                    break
        if entry:
            _render_code_diff(
                action="Update",
                display_path=display_path,
                pre_text=entry.pre_text,
                post_text=entry.post_text,
                old_snippet=entry.old_snippet,
                new_snippet="",
                old_start_line=entry.old_start_line,
            )
        else:
            _render_code_diff(
                action="Update",
                display_path=display_path,
                old_snippet=args.get("snippet", ""),
                new_snippet="",
            )

    elif name == "delete_file":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Delete[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")
        console.print("  [dim]└ Removed file[/dim]")

    elif name == "add_file":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Create[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")
        console.print("  [dim]└ Added 0 lines (empty file)[/dim]")

    elif name == "create_directory":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Directory[/bold][dim]([/dim][bold]{escape(display_path)}[/bold][dim])[/dim]")

    elif name == "read_file":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Read[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    elif name == "read_lines":
        s = args.get("start_line", 1)
        e = args.get("end_line", 100)
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Read[/bold][dim]([/dim]{escape(display_path)}:{s}-{e}[dim])[/dim]")

    elif name == "list_files":
        subdir = args.get("subdir", ".") or "."
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]List[/bold][dim]([/dim]{escape(str(subdir))}[dim])[/dim]")

    elif name == "grep_search":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Grep[/bold][dim]([/dim]{escape(args.get('query', ''))}[dim])[/dim]")

    elif name == "get_outline":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Outline[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    elif name == "run_command":
        cmd = args.get("command", "?")
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Run[/bold][dim]([/dim]{escape(cmd)}[dim])[/dim]")
        output_lines = result.strip().split("\n")
        if output_lines and output_lines != [""]:
            for line in output_lines[:15]:
                console.print(f"  [dim]│ {line}[/dim]")
            if len(output_lines) > 15:
                console.print(f"  [dim]│ ... ({len(output_lines) - 15} more lines)[/dim]")

    elif name == "fetch_url":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]Fetch[/bold][dim]([/dim]{escape(args.get('url', ''))}[dim])[/dim]")

    elif name == "search_web":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]WebSearch[/bold][dim]([/dim]{escape(args.get('query', ''))}[dim])[/dim]")

    elif name == "git_status":
        console.print("[bold #10b981]●[/bold #10b981] [bold]GitStatus[/bold]()")

    elif name == "git_diff":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]GitDiff[/bold][dim]([/dim]{escape(args.get('path', 'all'))}[dim])[/dim]")

    elif name == "git_commit":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]GitCommit[/bold][dim]([/dim]{escape(args.get('message', ''))}[dim])[/dim]")

    elif name == "validate_syntax":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]SyntaxCheck[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    elif name == "analyze_media":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]MediaAnalysis[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    elif name == "describe_image":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]VisionDescribe[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    elif name == "analyze_video":
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]VideoAnalysis[/bold][dim]([/dim]{escape(display_path)}[dim])[/dim]")

    else:
        console.print(f"[bold #10b981]●[/bold #10b981] [bold]{name}[/bold]()")


class Spinner:
    """
    A lightweight threaded terminal spinner for 'thinking' / 'learning'
    animations. Runs on a background thread writing \\r frames; stop() clears
    the line. Safe to stop multiple times.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str, interval: float = 0.1) -> None:
        import threading

        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional["object"] = None

    def start(self) -> "Spinner":
        import threading

        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        import itertools

        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{frame} {self.label} ")
            sys.stdout.flush()
            time.sleep(self.interval)

    def stop(self, clear: bool = True) -> None:
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass
        self._thread = None
        if clear:
            sys.stdout.write("\r" + " " * (len(self.label) + 4) + "\r")
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _type_out(text: str, cps: int = 120) -> None:
    """Print text to stdout with a typewriter animation (chars per second)."""
    import time as _time

    delay = 1.0 / cps if cps and cps > 0 else 0.0
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if delay:
            _time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _approval_prompt(action: str, detail: str) -> bool:
    """Show a proposed state-changing action and ask the user to approve it."""
    console.rule(f"Approve: {action}")
    console.print(detail)
    console.rule()
    answer = Prompt.ask("Allow this change?", choices=["y", "n"], default="n")
    return answer == "y"


def _ask_streaming(
    agent: Agent,
    prompt: str,
    ui_state: dict,
    *,
    on_tool_call=None,
    on_approval=None,
    thinking_label: str = "Thinking",
    header: str = "Agent",
) -> str:
    """
    agent.ask() with live token streaming: a spinner until the first token,
    then characters appear as they arrive. If the backend doesn't actually
    stream, the finished reply is typewriter-animated instead.
    """
    streamed: list[str] = []
    delay = 1.0 / ui_state["cps"] if (ui_state["typing"] and ui_state["cps"] > 0) else 0.0
    thinking = Spinner(thinking_label).start()

    def show_token(token: str) -> None:
        thinking.stop()
        if not streamed:
            console.print(f"[bold magenta]{header}[/bold magenta]: ", end="")
        streamed.append(token)
        if delay:
            for ch in token:
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(delay)
        else:
            sys.stdout.write(token)
            sys.stdout.flush()

    def wrapped_tool_call(name: str, args: dict, result: str) -> None:
        thinking.stop()
        if on_tool_call:
            on_tool_call(name, args, result)

    try:
        reply = agent.ask(
            prompt,
            on_tool_call=wrapped_tool_call,
            on_approval=on_approval,
            on_token=show_token,
        )
    except KeyboardInterrupt:
        partial = "".join(streamed)
        agent.recover_from_interruption(partial)
        console.print("\n[yellow]Interrupted.[/yellow]")
        return partial or "(interrupted)"
    finally:
        thinking.stop()

    if streamed:
        console.print()
    else:
        console.print(f"\n[bold magenta]{header}[/bold magenta]: ", end="")
        if ui_state["typing"]:
            _type_out(reply, ui_state["cps"])
        else:
            console.print(reply, markup=False, highlight=False)
    return reply


def _print_status_line(agent: Agent, cfg) -> None:
    """One-line footer shown right above the prompt: mode, model, tokens, flags."""
    flags = []
    if agent.plan_mode:
        flags.append("plan")
    if agent.require_approval:
        flags.append("approval")
    flag_str = f" · [yellow]{'+'.join(flags)}[/yellow]" if flags else ""
    total = agent.total_tokens_in + agent.total_tokens_out
    console.print(
        f"[dim]{agent.mode} · {agent.model.model_name} · "
        f"~{total:,} tokens · {len(agent.history)} msgs[/dim]{flag_str}"
    )


def _print_banner(cfg, agent: Agent, skill_registry, agent_memory, custom_commands) -> None:
    """Boxed welcome screen summarizing the session."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    grid.add_row("model", f"[bold]{cfg.model_name}[/bold]")
    if cfg.routing_mode == "hybrid":
        grid.add_row(
            "routing",
            f"hybrid · fast [bold]{cfg.fast_model or cfg.model_name}[/bold] · "
            f"reasoning [bold]{cfg.reasoning_model or cfg.model_name}[/bold]",
        )
    grid.add_row("project", str(cfg.project_root))
    grid.add_row(
        "skills",
        f"{len(skill_registry)} loaded" if skill_registry else "[yellow]none (use --skills-path)[/yellow]",
    )
    grid.add_row(
        "memory",
        f"{len(agent_memory.lessons)} lessons" if not agent_memory.is_empty() else "empty",
    )
    if agent.project_instructions:
        grid.add_row("instructions", "CLAUDE.md / AGENTS.md loaded")
    if custom_commands:
        grid.add_row("commands", ", ".join(f"/{n}" for n in custom_commands))
    grid.add_row(
        "limits",
        f"{cfg.max_iterations} iterations · {cfg.max_context_turns} turns · {cfg.command_timeout}s commands",
    )
    console.print(
        Panel(
            grid,
            title="[bold green]AI Agent[/bold green]",
            subtitle="[dim]/help for commands · Tab completes /commands · /exit to quit[/dim]",
            border_style="green",
            expand=False,
            padding=(1, 2),
        )
    )


def _setup_readline(history_file: Path, commands: list[str]) -> None:
    """Persist prompt history across sessions and Tab-complete slash commands."""
    if readline is None:
        return
    try:
        if history_file.exists():
            readline.read_history_file(str(history_file))
        readline.set_history_length(1000)
    except (OSError, ValueError):
        pass
    atexit.register(_save_history, history_file)

    options = sorted(set(commands))

    def completer(text: str, state: int):
        matches = [c for c in options if c.startswith(text)] if text.startswith("/") else []
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _save_history(history_file: Path) -> None:
    try:
        readline.write_history_file(str(history_file))
    except OSError:
        pass


SLASH_COMMANDS = [
    "/agent", "/approve", "/attach", "/audio", "/auto-skills", "/autocorrect", "/autonomous",
    "/best", "/changes", "/clear", "/clear-skills", "/commit", "/compact", "/continuous", "/cost",
    "/data", "/delete", "/delete-session", "/diff", "/doctor", "/exit", "/export", "/export-sft", "/forget",
    "/git", "/go", "/grep", "/help", "/hooks", "/http", "/image", "/images", "/init",
    "/iterate", "/limits", "/lines", "/list", "/loop", "/mcp", "/media", "/memory", "/mkdir",
    "/mode", "/model", "/models", "/modes", "/new", "/outline", "/paste", "/patch", "/plan",
    "/read", "/reasoning", "/refine", "/reflect", "/remember", "/rename", "/resume", "/review",
    "/routing", "/save", "/search", "/sessions", "/skills", "/status", "/stream", "/syntax",
    "/task", "/tasks", "/test", "/todo", "/tokens", "/typing", "/undo", "/upload", "/use",
    "/video", "/videos", "/write",
]


def _print_skills(registry: SkillRegistry, query: str | None = None) -> None:
    if not registry:
        console.print("[yellow]No skills found.[/yellow]")
        return

    skills = registry.match(query, limit=20) if query else registry.all()
    if not skills:
        console.print("[yellow]No matching skills found.[/yellow]")
        return

    table = Table(title="Available Skills" if not query else f"Skills matching: {query}")
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Description")

    for skill in skills:
        description = skill.description.replace("\n", " ")
        if len(description) > 120:
            description = description[:117].rstrip() + "..."
        table.add_row(skill.name, description)

    console.print(table)


def _print_modes(agent: Agent) -> None:
    table = Table(title="Agent Modes")
    table.add_column("Mode", style="bold cyan", no_wrap=True)
    table.add_column("Active", justify="center")
    for mode in agent.available_modes():
        table.add_row(mode, "✓" if mode == agent.mode else "")
    console.print(table)


def _print_limits(cfg) -> None:
    table = Table(title="Capacity Settings")
    table.add_column("Setting", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("max_context_turns", str(cfg.max_context_turns))
    table.add_row("max_iterations", str(cfg.max_iterations))
    table.add_row("max_auto_skills", str(cfg.max_auto_skills))
    table.add_row("skill_prompt_chars", str(cfg.skill_prompt_chars))
    table.add_row("command_timeout", f"{cfg.command_timeout}s")
    table.add_row(
        "list_output_limit",
        "unlimited" if cfg.list_output_limit <= 0 else str(cfg.list_output_limit),
    )
    table.add_row("include_ignored_files", str(cfg.include_ignored_files))
    console.print(table)


def _print_changes(tracker: ChangeTracker) -> None:
    """Display a rich table of all recorded changes."""
    stats = tracker.stats()
    if stats:
        table = Table(title="Session Changes — Summary")
        table.add_column("Action", style="bold", no_wrap=True)
        table.add_column("Count", justify="right")
        color_map = {
            "create": "green",
            "write": "yellow",
            "patch": "blue",
            "delete_file": "red",
            "delete_code": "red",
            "mkdir": "cyan",
        }
        for action, count in sorted(stats.items()):
            color = color_map.get(action, "white")
            table.add_row(f"[{color}]{action}[/{color}]", str(count))
        console.print(table)

    if not tracker.entries:
        console.print("[dim]No changes recorded yet.[/dim]")
        return

    detail_table = Table(title="Session Changes — Detail", show_lines=True)
    detail_table.add_column("Time", style="dim", no_wrap=True)
    detail_table.add_column("Action", no_wrap=True)
    detail_table.add_column("File")
    detail_table.add_column("Details", style="dim")

    color_map = {
        "create": "green",
        "write": "yellow",
        "patch": "blue",
        "delete_file": "red",
        "delete_code": "red",
        "mkdir": "cyan",
    }
    import time as _time
    for entry in tracker.entries:
        t = _time.strftime("%H:%M:%S", _time.localtime(entry.timestamp))
        color = color_map.get(entry.action, "white")
        detail_table.add_row(
            t,
            f"[{color}]{entry.action}[/{color}]",
            entry.path,
            entry.details,
        )
    console.print(detail_table)


def _read_multiline(label: str) -> str:
    console.print(f"[bold]{label}[/bold] Finish with a line containing EOF.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)


@app.command()
def main(
    project_path: Optional[Path] = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Path to the project the agent should work on.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (e.g. llama3.1:8b, commandcode:deepseek/deepseek-v4-pro, "
        "claude-3-5-sonnet-20241022; or any name if custom_model_url is set).",
    ),
    max_turns: Optional[int] = typer.Option(
        None,
        "--max-turns",
        help="Maximum number of recent turns to keep in context.",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Maximum agent tool/reasoning loop iterations.",
    ),
    max_auto_skills: Optional[int] = typer.Option(
        None,
        "--max-auto-skills",
        help="Maximum automatically matched skills per request.",
    ),
    skill_prompt_chars: Optional[int] = typer.Option(
        None,
        "--skill-prompt-chars",
        help="Maximum characters injected from each skill.",
    ),
    command_timeout: Optional[int] = typer.Option(
        None,
        "--command-timeout",
        help="Shell command timeout in seconds.",
    ),
    list_output_limit: Optional[int] = typer.Option(
        None,
        "--list-output-limit",
        help="Maximum files shown by /list. Use 0 for unlimited.",
    ),
    include_ignored_files: bool = typer.Option(
        False,
        "--include-ignored-files/--no-include-ignored-files",
        help="Include noisy folders such as .git, .venv, and __pycache__ in file listings.",
    ),
    reasoning: Optional[bool] = typer.Option(
        None,
        "--reasoning/--no-reasoning",
        help="Enable Ollama's native reasoning pass for supported local models.",
    ),
    skills_path: Optional[Path] = typer.Option(
        None,
        "--skills-path",
        help="Path to a directory containing Claude-style skills.",
    ),
) -> None:
    """
    Start an interactive REPL with the local AI agent.
    """
    # Default to this script's own directory, NOT the current working
    # directory. cli.py is sometimes launched by double-clicking it directly
    # (bypassing start_cli.bat, which does `cd` first) or from a shortcut
    # whose working directory can be anywhere — even a protected folder like
    # C:\Windows\System32, where the agent then fails trying to create
    # .agent_sessions/.agent_memory.json there. See webui.py's identical fix.
    cfg_project_root = project_path.resolve() if project_path else Path(__file__).resolve().parent
    cfg_skills_root = skills_path.resolve() if skills_path else None
    cfg = load_config(project_root=cfg_project_root, skills_root=cfg_skills_root)

    if model:
        cfg.model_name = model
    if max_turns is not None:
        cfg.max_context_turns = max_turns
    if max_iterations is not None:
        cfg.max_iterations = max_iterations
    if max_auto_skills is not None:
        cfg.max_auto_skills = max_auto_skills
    if skill_prompt_chars is not None:
        cfg.skill_prompt_chars = skill_prompt_chars
    if command_timeout is not None:
        cfg.command_timeout = command_timeout
    if list_output_limit is not None:
        cfg.list_output_limit = list_output_limit
    if reasoning is not None:
        cfg.reasoning_enabled = reasoning
    cfg.include_ignored_files = include_ignored_files

    tools = ProjectTools(
        project_root=cfg.project_root,
        command_timeout=cfg.command_timeout,
        include_ignored_files=cfg.include_ignored_files,
        restrict_to_project=cfg.restrict_to_project,
    )
    model_client = ModelClient(
        model_name=cfg.model_name,
        timeout=cfg.request_timeout,
        reasoning=cfg.reasoning_enabled,
        routing_mode=cfg.routing_mode,
        fast_model=cfg.fast_model,
        reasoning_model=cfg.reasoning_model,
    )
    skill_registry = SkillRegistry(cfg.skills_root)
    memory_path = cfg.memory_path or (cfg.project_root / ".agent_memory.json")
    agent_memory = AgentMemory.load(memory_path, max_lessons=cfg.memory_max_lessons)
    settings = load_settings(cfg.project_root)
    hook_manager = HookManager(settings["hooks"], cfg.project_root)
    agent = Agent(
        model=model_client,
        tools=tools,
        skills=skill_registry,
        mode=cfg.mode,
        max_context_turns=cfg.max_context_turns,
        max_iterations=cfg.max_iterations,
        max_auto_skills=cfg.max_auto_skills,
        skill_prompt_chars=cfg.skill_prompt_chars,
        mcp=MCPClientManager(cfg.mcp_servers, timeout=cfg.request_timeout),
        memory=agent_memory,
        compact_threshold=cfg.compact_threshold,
        project_instructions=_load_project_instructions(cfg.project_root),
        hooks=hook_manager,
    )
    
    # Initialize autonomous agent with advanced capabilities
    autonomous_config = AutonomousConfig(
        enable_auto_continuation=True,
        enable_iteration=True,
        enable_loops=True,
        enable_task_decomposition=True,
        enable_self_correction=True,
        enable_streaming=True,
    )
    autonomous = AutonomousAgent(agent, autonomous_config)

    def reload_hooks() -> None:
        """Re-read .agent/settings.json into the live agent."""
        fresh = load_settings(cfg.project_root)
        agent.hooks = HookManager(fresh["hooks"], cfg.project_root)

    session_store = SessionStore(cfg.project_root / ".agent_sessions")
    custom_commands = _load_custom_commands(cfg.project_root)
    # Typewriter animation settings for the agent's output.
    ui_state = {"typing": True, "cps": 120}

    if not cfg.project_root.exists():
        console.print(
            f"[red]Project path does not exist:[/red] {cfg.project_root}\n"
            "Start again with --project-path C:/path/to/your/project or update config.yaml."
        )
        raise typer.Exit(code=1)

    saved_sessions = session_store.list_sessions()
    current_session_id = uuid4().hex[:12]
    current_session_title = "New chat"
    if saved_sessions:
        latest = saved_sessions[0]
        try:
            data = session_store.load(str(latest["id"]))
            stored_history = data.get("history")
            if isinstance(stored_history, list) and all(isinstance(item, dict) for item in stored_history):
                agent.history = stored_history
                current_session_id = str(data.get("id", latest["id"]))
                current_session_title = str(data.get("title") or "New chat")
                agent.tasks = data.get("tasks", []) if isinstance(data.get("tasks", []), list) else []
                if data.get("mode") in agent.available_modes():
                    agent.mode = data["mode"]
                agent.plan_mode = bool(data.get("plan_mode", False))
                agent.require_approval = bool(data.get("require_approval", False))
                stored_model = str(data.get("model_name") or "").strip()
                if stored_model and not model:
                    cfg.model_name = stored_model
                    model_client = ModelClient(
                        model_name=stored_model,
                        timeout=cfg.request_timeout,
                        reasoning=cfg.reasoning_enabled,
                        routing_mode=cfg.routing_mode,
                        fast_model=cfg.fast_model,
                        reasoning_model=cfg.reasoning_model,
                    )
                    agent.model = model_client
                console.print(
                    f"[green]Resumed conversation[/green] {current_session_title} "
                    f"(id: {current_session_id})."
                )
        except (OSError, ValueError, KeyError):
            pass

    def save_active_session() -> None:
        nonlocal current_session_title
        if current_session_title == "New chat":
            first_user = next(
                (message.get("content") for message in agent.history if message.get("role") == "user"),
                None,
            )
            if first_user:
                current_session_title = " ".join(str(first_user).split())[:64]
        session_store.save(
            current_session_id,
            agent.history,
            agent.mode,
            agent.tasks,
            title=current_session_title,
            plan_mode=agent.plan_mode,
            require_approval=agent.require_approval,
            model_name=model_client.model_name,
        )

    def set_active_model(model_name: str) -> None:
        nonlocal model_client
        model_client = ModelClient(
            model_name=model_name,
            timeout=cfg.request_timeout,
            reasoning=cfg.reasoning_enabled,
            routing_mode=cfg.routing_mode,
            fast_model=cfg.fast_model,
            reasoning_model=cfg.reasoning_model,
        )
        cfg.model_name = model_name
        agent.model = model_client

    if not saved_sessions:
        save_active_session()

    _print_banner(cfg, agent, skill_registry, agent_memory, custom_commands)
    _setup_readline(
        cfg.project_root / ".agent_repl_history",
        SLASH_COMMANDS + [f"/{n}" for n in custom_commands],
    )
    _print_help()

    for name, status in agent.mcp.status().items():
        if status.get("connected"):
            console.print(
                f"[green]MCP connected:[/green] {name} "
                f"({status.get('tools', 0)} tools)"
            )
        else:
            console.print(f"[yellow]MCP connection issue:[/yellow] {name}: {status.get('error')}")

    while True:
        try:
            _print_status_line(agent, cfg)
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting.")
            break

        stripped = user_input.strip().lstrip("\ufeff")
        if not stripped:
            continue

        if stripped in {"/exit", ":q"}:
            save_active_session()
            console.print("Goodbye.")
            break

        if stripped in {"/help", ":h"}:
            _print_help()
            continue

        if stripped == "/hooks" or stripped == "/hooks reload":
            if stripped.endswith("reload"):
                reload_hooks()
                console.print(f"[green]Reloaded[/green] {settings_path(cfg.project_root)}")
            lines = agent.hooks.describe() if agent.hooks else []
            if not lines:
                console.print("[dim]No hooks configured.[/dim]")
                console.print(
                    f"Add them to {settings_path(cfg.project_root)}, then /hooks reload. Example:\n"
                    '{"hooks": {"PreToolUse": [{"matcher": "run_command", '
                    '"command": "python check.py"}]}}\n'
                    "Events: PreToolUse, PostToolUse, UserPromptSubmit, Stop. The hook gets JSON on "
                    "stdin; exit 2 blocks and its stderr goes back to the model.",
                    markup=False,
                )
            else:
                for line in lines:
                    console.print(line, markup=False, highlight=False)
            continue

        # "!cmd" runs a shell command directly.
        if stripped.startswith("!") and len(stripped) > 1:
            shell_cmd = stripped[1:].strip()
            console.print(f"$ {shell_cmd}", markup=False, highlight=False, style="dim")
            console.print(tools.run_command(shell_cmd), markup=False, highlight=False)
            continue

        # "#note" saves a lesson to persistent memory.
        if stripped.startswith("#") and len(stripped) > 1:
            note = stripped[1:].strip()
            if agent.memory and agent.memory.add_lesson(note):
                console.print(f"[green]Remembered:[/green] {escape(note)}")
            else:
                console.print("[yellow]Not saved (empty, duplicate, or memory disabled).[/yellow]")
            continue

        if stripped.startswith("/http "):
            url = stripped.split(maxsplit=1)[1]
            try:
                text = fetch_url(url)
                console.rule(f"Response from {url}")
                console.print(text)
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error fetching URL:[/red] {exc}")
            continue

        if stripped.startswith("/search "):
            query = stripped.split(maxsplit=1)[1]
            try:
                text = search_web(query)
                console.rule(f"Search results for: {query}")
                console.print(text)
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error searching web:[/red] {exc}")
            continue

        if stripped.startswith("/read "):
            path = stripped.split(maxsplit=1)[1]
            try:
                console.rule(path)
                console.print(tools.read_file(path))
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error reading file:[/red] {exc}")
            continue

        if stripped == "/list" or stripped.startswith("/list "):
            subdir = stripped.split(maxsplit=1)[1] if " " in stripped else None
            try:
                files = tools.list_files(subdir=subdir)
                if not files:
                    console.print("[yellow]No files found.[/yellow]")
                else:
                    shown_files = (
                        files
                        if cfg.list_output_limit <= 0
                        else files[: cfg.list_output_limit]
                    )
                    for path in shown_files:
                        console.print(path.relative_to(tools.project_root))
                    if cfg.list_output_limit > 0 and len(files) > cfg.list_output_limit:
                        console.print(
                            f"[dim]... {len(files) - cfg.list_output_limit} more files. "
                            "Use /list <subdir> to narrow the output.[/dim]"
                        )
            except Exception as exc:
                console.print(f"[red]Error listing files:[/red] {exc}")
            continue

        if stripped.startswith("/write "):
            path = stripped.split(maxsplit=1)[1]
            content = _read_multiline("Enter file content.")
            try:
                tools.write_file(path, content)
                console.print(f"[green]Wrote file:[/green] {path}")
            except Exception as exc:
                console.print(f"[red]Error writing file:[/red] {exc}")
            continue

        if stripped.startswith("/patch "):
            path = stripped.split(maxsplit=1)[1]
            old_snippet = _read_multiline("Enter old snippet.")
            new_snippet = _read_multiline("Enter new snippet.")
            try:
                changed = tools.apply_patch(path, old_snippet, new_snippet)
                if changed:
                    console.print(f"[green]Patched file:[/green] {path}")
                else:
                    console.print("[yellow]Old snippet not found. No changes made.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error patching file:[/red] {exc}")
            continue

        if stripped.startswith("/delete "):
            path = stripped.split(maxsplit=1)[1]
            try:
                deleted = tools.delete_file(path)
                if deleted:
                    console.print(f"[green]Deleted file:[/green] {path}")
                else:
                    console.print(f"[yellow]File not found:[/yellow] {path}")
            except Exception as exc:
                console.print(f"[red]Error deleting file:[/red] {exc}")
            continue

        if stripped.startswith("/mkdir "):
            path = stripped.split(maxsplit=1)[1]
            try:
                result_path = tools.create_directory(path)
                console.print(f"[green]Created directory:[/green] {result_path}")
            except Exception as exc:
                console.print(f"[red]Error creating directory:[/red] {exc}")
            continue

        if stripped == "/changes" or stripped.startswith("/changes "):
            sub = stripped.split(maxsplit=1)[1] if " " in stripped else None
            if sub == "clear":
                tools.changes.clear()
                console.print("[green]Changes log cleared.[/green]")
            else:
                _print_changes(tools.changes)
            continue

        if stripped == "/skills" or stripped.startswith("/skills "):
            query = stripped.split(maxsplit=1)[1] if " " in stripped else None
            _print_skills(skill_registry, query=query)
            continue

        if stripped.startswith("/use "):
            skill_name = stripped.split(maxsplit=1)[1]
            try:
                skill = agent.activate_skill(skill_name)
                console.print(f"[green]Loaded skill:[/green] {skill.name}")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        if stripped == "/clear-skills":
            agent.clear_skills()
            console.print("[green]Manual skills cleared.[/green]")
            continue

        if stripped.startswith("/auto-skills"):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 1:
                state = "on" if agent.auto_use_skills else "off"
                console.print(f"Auto skill matching is [bold]{state}[/bold].")
                continue
            value = parts[1].lower()
            if value not in {"on", "off"}:
                console.print("[red]Use /auto-skills on or /auto-skills off.[/red]")
                continue
            agent.auto_use_skills = value == "on"
            console.print(f"[green]Auto skill matching {value}.[/green]")
            continue

        if stripped == "/modes":
            _print_modes(agent)
            continue

        if stripped == "/models":
            try:
                models = model_client.list_available_models()
                console.print("[bold]Available models[/bold]")
                for model_name in models:
                    marker = " *" if model_name == model_client.model_name else ""
                    console.print(f"  {model_name}{marker}")
            except Exception as exc:
                console.print(f"[yellow]Could not list provider models:[/yellow] {exc}")
            continue

        if stripped == "/model" or stripped.startswith("/model "):
            if stripped == "/model":
                console.print(f"Current model: [bold]{model_client.model_name}[/bold]")
                continue
            selected_model = stripped.split(maxsplit=1)[1].strip()
            if not selected_model:
                console.print("Use /model <name>.")
                continue
            set_active_model(selected_model)
            save_active_session()
            console.print(f"[green]Model switched to:[/green] {selected_model}")
            continue

        if stripped == "/reasoning" or stripped.startswith("/reasoning "):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 1:
                state = "on" if model_client.reasoning else "off"
                console.print(f"Native reasoning is [bold]{state}[/bold].")
                continue
            value = parts[1].strip().lower()
            if value not in {"on", "off"}:
                console.print("[yellow]Use /reasoning on or /reasoning off.[/yellow]")
                continue
            cfg.reasoning_enabled = value == "on"
            model_client.reasoning = cfg.reasoning_enabled
            console.print(f"[green]Native reasoning {value}.[/green]")
            continue

        if stripped == "/routing":
            console.print(
                f"Routing: [bold]{model_client.routing_mode}[/bold]\n"
                f"Fast model: [bold]{model_client.fast_model or model_client.model_name}[/bold]\n"
                f"Reasoning model: [bold]{model_client.reasoning_model or model_client.model_name}[/bold]\n"
                f"Last selected: [bold]{model_client.last_routed_model or 'none yet'}[/bold]"
            )
            continue

        if stripped == "/limits":
            _print_limits(cfg)
            continue

        if stripped == "/mcp":
            statuses = agent.mcp.status() if agent.mcp else {}
            if not statuses:
                console.print("No MCP servers configured. Add entries under mcp_servers in config.yaml.")
            for name, status in statuses.items():
                state = "connected" if status.get("connected") else "disconnected"
                color = "green" if status.get("connected") else "yellow"
                console.print(
                    f"[{color}]{name}: {state}[/{color}] — "
                    f"{status.get('tools', 0)} tools"
                )
                if status.get("error"):
                    console.print(f"  [dim]{status['error']}[/dim]")
            continue

        if stripped == "/mode" or stripped.startswith("/mode "):
            if stripped == "/mode":
                console.print(f"Current mode: [bold]{agent.mode}[/bold]")
                continue
            mode_name = stripped.split(maxsplit=1)[1]
            try:
                agent.set_mode(mode_name)
                save_active_session()
                console.print(f"[green]Mode switched to:[/green] {agent.mode}")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        if stripped == "/diff" or stripped.startswith("/diff "):
            target_file = stripped.split(maxsplit=1)[1] if " " in stripped else None
            diff_text = tools.git_diff(file_path=target_file)
            console.rule("Git Diff")
            console.print(diff_text)
            console.rule()
            continue

        if stripped.startswith("/git "):
            subcmd = stripped.split(maxsplit=1)[1]
            output = tools.run_command(f"git {subcmd}")
            console.rule(f"git {subcmd}")
            console.print(output)
            console.rule()
            continue

        if stripped == "/undo" or stripped.startswith("/undo "):
            path = stripped.split(maxsplit=1)[1] if " " in stripped else None
            res = tools.undo_last_change(path)
            console.print(f"[yellow]{res}[/yellow]")
            continue

        if stripped.startswith("/grep "):
            query = stripped.split(maxsplit=1)[1]
            res = tools.grep_search(query)
            console.rule(f"Grep: {query}")
            console.print(res)
            console.rule()
            continue

        if stripped.startswith("/outline "):
            path = stripped.split(maxsplit=1)[1]
            res = tools.get_outline(path)
            console.rule(f"Outline: {path}")
            console.print(res)
            console.rule()
            continue

        if stripped.startswith("/lines "):
            parts = stripped.split()
            path = parts[1]
            s = int(parts[2]) if len(parts) > 2 else 1
            e = int(parts[3]) if len(parts) > 3 else 100
            try:
                res = tools.read_lines(path, start_line=s, end_line=e)
                console.rule(f"{path} ({s}-{e})")
                console.print(res)
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
            continue

        if stripped.startswith("/syntax "):
            path = stripped.split(maxsplit=1)[1]
            res = tools.validate_syntax(path)
            color = "green" if "OK" in res else "red"
            console.print(f"[{color}]{res}[/{color}]")
            continue

        if stripped == "/test" or stripped.startswith("/test "):
            cmd = stripped.split(maxsplit=1)[1] if " " in stripped else "pytest -v"
            console.print(f"[dim]Running test: {cmd}[/dim]")
            res = tools.run_command(cmd)
            console.rule(cmd)
            console.print(res)
            console.rule()
            continue

        if stripped == "/compact":
            with Spinner("Compressing context into memory"):
                res = agent.compact()
            console.print(f"[green]{res}[/green]")
            continue

        if stripped == "/clear":
            dropped = agent.clear_history()
            console.print(f"[green]Context cleared.[/green] Dropped {dropped} message(s). "
                          "Memory, tasks, skills, and mode kept.")
            continue

        if stripped == "/init":
            console.print("[dim]Scanning project to generate CLAUDE.md...[/dim]")
            try:
                files = tools.list_files()
                rels = [str(p.relative_to(tools.project_root)) for p in files]
                listing = "\n".join(rels[:200])
                if len(rels) > 200:
                    listing += f"\n...(+{len(rels) - 200} more files)"
                # Pull snippets from a few telling top-level files.
                snippet_targets = [
                    "README.md", "readme.md", "pyproject.toml", "package.json",
                    "requirements.txt", "Cargo.toml", "go.mod", "config.yaml",
                ]
                snippets = []
                for name in snippet_targets:
                    path = tools.project_root / name
                    if path.is_file():
                        try:
                            body = path.read_text(encoding="utf-8")[:1500]
                        except OSError:
                            continue
                        snippets.append(f"--- {name} ---\n{body}")
                snippet_text = "\n\n".join(snippets) or "(no manifest/readme files found)"
                content = agent.generate_project_instructions(listing, snippet_text)
                target = tools.project_root / "CLAUDE.md"
                if target.exists():
                    answer = Prompt.ask(
                        f"{target.name} exists. Overwrite?", choices=["y", "n"], default="n"
                    )
                    if answer != "y":
                        console.print("[yellow]Kept existing CLAUDE.md. Not written.[/yellow]")
                        continue
                tools.write_file("CLAUDE.md", content)
                agent.project_instructions = _load_project_instructions(cfg.project_root)
                console.print(f"[green]Wrote {target} and loaded it as project instructions.[/green]")
            except Exception as exc:
                console.print(f"[red]Error during /init:[/red] {exc}")
            continue

        if stripped == "/memory":
            mem = agent.memory
            if mem is None or mem.is_empty():
                console.print("[dim]Persistent memory is empty.[/dim]")
            else:
                if mem.rolling_summary:
                    console.rule("Rolling summary")
                    console.print(mem.rolling_summary)
                if mem.lessons:
                    console.rule(f"Lessons ({len(mem.lessons)})")
                    for i, lesson in enumerate(mem.lessons, 1):
                        console.print(f"  {i}. {lesson}")
                console.print(f"[dim]Stored at {mem.file_path}[/dim]")
            continue

        if stripped.startswith("/remember "):
            note = stripped.split(maxsplit=1)[1]
            if agent.memory and agent.memory.add_lesson(note):
                console.print(f"[green]Remembered:[/green] {note}")
            else:
                console.print("[yellow]Not saved (empty, duplicate, or memory disabled).[/yellow]")
            continue

        if stripped == "/reflect":
            with Spinner("Learning from this session"):
                res = agent.reflect()
            console.print(f"[green]{res}[/green]")
            continue

        if stripped == "/forget" or stripped.startswith("/forget "):
            arg = stripped.split(maxsplit=1)[1].strip() if " " in stripped else None
            if agent.memory is None:
                console.print("[yellow]Memory is disabled.[/yellow]")
            elif arg == "all":
                agent.memory.clear()
                console.print("[green]Cleared all persistent memory.[/green]")
            elif arg and arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(agent.memory.lessons):
                    removed = agent.memory.lessons.pop(idx)
                    agent.memory.save()
                    console.print(f"[green]Forgot lesson:[/green] {removed}")
                else:
                    console.print("[yellow]No lesson with that number. See /memory.[/yellow]")
            else:
                console.print("[yellow]Use /forget all or /forget <number> (see /memory).[/yellow]")
            continue

        if stripped == "/plan" or stripped.startswith("/plan "):
            arg = stripped.split(maxsplit=1)[1].lower().strip() if " " in stripped else None
            if arg == "on":
                agent.plan_mode = True
                console.print("[green]Plan mode ON — agent will plan, not edit.[/green]")
            elif arg == "off":
                agent.plan_mode = False
                console.print("[green]Plan mode OFF — agent may make changes.[/green]")
            else:
                console.print(f"Plan mode is [bold]{'on' if agent.plan_mode else 'off'}[/bold]. Use /plan on|off.")
            save_active_session()
            continue

        if stripped == "/approve" or stripped.startswith("/approve "):
            arg = stripped.split(maxsplit=1)[1].lower().strip() if " " in stripped else None
            if arg == "on":
                agent.require_approval = True
                console.print("[green]Edit approval ON — you confirm each change.[/green]")
            elif arg == "off":
                agent.require_approval = False
                console.print("[green]Edit approval OFF — changes apply automatically.[/green]")
            else:
                state = "on" if agent.require_approval else "off"
                console.print(f"Edit approval is [bold]{state}[/bold]. Use /approve on|off.")
            save_active_session()
            continue

        if stripped == "/todo" or stripped == "/tasks":
            console.rule("Task list")
            console.print(agent.render_tasks())
            console.rule()
            continue

        if stripped.startswith("/task "):
            rest = stripped.split(maxsplit=1)[1]
            parts = rest.split(maxsplit=2)
            if parts[0] == "add" and len(parts) > 1:
                task = agent.add_task(rest.split(maxsplit=1)[1])
                console.print(f"[green]Added task {task['id']}:[/green] {task['text']}")
            elif parts[0] == "done" and len(parts) > 1 and parts[1].isdigit():
                ok = agent.update_task(int(parts[1]), "done")
                console.print("[green]Marked done.[/green]" if ok else "[yellow]No such task.[/yellow]")
            elif parts[0] == "clear":
                agent.clear_tasks()
                console.print("[green]Task list cleared.[/green]")
            else:
                console.print("[yellow]Use: /task add <text> | /task done <n> | /task clear[/yellow]")
            save_active_session()
            continue

        if stripped.startswith("/save") :
            name = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
            session_id = name or current_session_id
            path = session_store.save(
                session_id,
                agent.history,
                agent.mode,
                agent.tasks,
                title=name or current_session_title,
                plan_mode=agent.plan_mode,
                require_approval=agent.require_approval,
                model_name=model_client.model_name,
            )
            console.print(f"[green]Session saved:[/green] {path.stem} ({path})")
            continue

        if stripped == "/sessions":
            rows = session_store.list_sessions()
            if not rows:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                table = Table(title="Saved Sessions")
                table.add_column("ID", style="bold cyan")
                table.add_column("Title")
                table.add_column("Mode")
                table.add_column("Model")
                table.add_column("Messages", justify="right")
                table.add_column("Saved", style="dim")
                import time as _t
                for r in rows:
                    ts = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(r["saved"]))
                    table.add_row(
                        r["id"], r.get("title", "New chat"), r["mode"],
                        r.get("model_name", ""), str(r["messages"]), ts,
                    )
                console.print(table)
                console.print(f"[dim]Active session: {current_session_id}[/dim]")
            continue

        if stripped == "/go" or stripped.startswith("/go "):
            arg = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
            rows = session_store.list_sessions()
            if not rows:
                console.print("[yellow]No saved sessions to go to.[/yellow]")
                continue
            target_sid = None
            if not arg:
                table = Table(title="Go to session")
                table.add_column("#", style="bold cyan", justify="right")
                table.add_column("Title")
                table.add_column("Mode")
                table.add_column("Msgs", justify="right")
                table.add_column("ID", style="dim")
                for i, r in enumerate(rows, 1):
                    table.add_row(str(i), str(r.get("title", "")), str(r.get("mode", "?")),
                                  str(r.get("messages", 0)), str(r.get("id", "")))
                console.print(table)
                choice = Prompt.ask("Go to which # (blank to cancel)", default="")
                choice = choice.strip()
                if not choice:
                    continue
                if choice.isdigit() and 1 <= int(choice) <= len(rows):
                    target_sid = str(rows[int(choice) - 1]["id"])
                else:
                    console.print("[yellow]Invalid number.[/yellow]")
                    continue
            elif arg.isdigit() and 1 <= int(arg) <= len(rows):
                target_sid = str(rows[int(arg) - 1]["id"])
            else:
                target_sid = arg  # treat as a session id
            try:
                save_active_session()  # keep the current chat before switching
                data = session_store.load(target_sid)
                agent.history = data.get("history") or agent.history
                agent.tasks = data.get("tasks") or []
                if data.get("mode"):
                    agent.mode = data["mode"]
                agent.plan_mode = bool(data.get("plan_mode", False))
                agent.require_approval = bool(data.get("require_approval", False))
                stored_model = str(data.get("model_name") or "").strip()
                if stored_model:
                    set_active_model(stored_model)
                current_session_id = str(data.get("id", target_sid))
                current_session_title = str(data.get("title") or "New chat")
                console.print(
                    f"[green]Switched to session {current_session_id}[/green] "
                    f"({len(agent.history)} messages, mode {agent.mode})."
                )
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        if stripped.startswith("/resume "):
            sid = stripped.split(maxsplit=1)[1].strip()
            try:
                data = session_store.load(sid)
                agent.history = data.get("history") or agent.history
                agent.tasks = data.get("tasks") or []
                if data.get("mode"):
                    agent.mode = data["mode"]
                agent.plan_mode = bool(data.get("plan_mode", False))
                agent.require_approval = bool(data.get("require_approval", False))
                stored_model = str(data.get("model_name") or "").strip()
                if stored_model:
                    set_active_model(stored_model)
                current_session_id = str(data.get("id", sid))
                current_session_title = str(data.get("title") or "New chat")
                console.print(
                    f"[green]Resumed session {data.get('id', sid)}[/green] "
                    f"({len(agent.history)} messages, mode {agent.mode})."
                )
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        if stripped == "/new" or stripped.startswith("/new "):
            save_active_session()
            current_session_id = uuid4().hex[:12]
            current_session_title = stripped.partition(" ")[2].strip() or "New chat"
            agent.history = agent.history[:1]
            agent.tasks = []
            agent.plan_mode = False
            save_active_session()
            console.print(
                f"[green]Started conversation[/green] {current_session_title} "
                f"(id: {current_session_id})."
            )
            continue

        if stripped == "/typing" or stripped.startswith("/typing "):
            parts = stripped.split()
            if len(parts) == 1:
                state = "on" if ui_state["typing"] else "off"
                console.print(f"Typing animation: [bold]{state}[/bold] at {ui_state['cps']} chars/sec. "
                              "Use /typing on|off [cps].")
            else:
                arg = parts[1].lower()
                if arg == "on":
                    ui_state["typing"] = True
                elif arg == "off":
                    ui_state["typing"] = False
                if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) > 0:
                    ui_state["cps"] = int(parts[2])
                elif arg.isdigit() and int(arg) > 0:
                    ui_state["typing"] = True
                    ui_state["cps"] = int(arg)
                state = "on" if ui_state["typing"] else "off"
                console.print(f"[green]Typing animation {state}[/green] ({ui_state['cps']} chars/sec).")
            continue

        if stripped == "/status":
            table = Table(title="Session Status")
            table.add_column("Field", style="bold cyan", no_wrap=True)
            table.add_column("Value")
            table.add_row("model", agent.model.model_name)
            table.add_row("routing", agent.model.routing_mode)
            table.add_row("last routed model", agent.model.last_routed_model or "none yet")
            table.add_row("mode", agent.mode)
            table.add_row("plan_mode", "on" if agent.plan_mode else "off")
            table.add_row("approval", "on" if agent.require_approval else "off")
            table.add_row("messages", str(len(agent.history)))
            table.add_row("tasks", str(len(agent.tasks)))
            mem_lessons = len(agent.memory.lessons) if agent.memory else 0
            table.add_row("memory lessons", str(mem_lessons))
            table.add_row("active skills", str(len(agent.active_skill_names)))
            table.add_row(
                "est. tokens (in/out)",
                f"{agent.total_tokens_in:,} / {agent.total_tokens_out:,}",
            )
            console.print(table)
            continue

        if stripped == "/review" or stripped.startswith("/review "):
            target = stripped.split(maxsplit=1)[1].strip() if " " in stripped else None
            diff_text = tools.git_diff(file_path=target)
            if not diff_text.strip():
                console.print("[yellow]No diff to review (working tree clean?).[/yellow]")
                continue
            console.rule("Review")
            _ask_streaming(
                agent,
                "Review the following git diff as a senior reviewer. Prioritize bugs, "
                "regressions, security issues, and missing tests. Be specific about files "
                "and lines. Do not modify anything.\n\n```diff\n" + diff_text[:12000] + "\n```",
                ui_state,
                on_tool_call=_on_tool_call,
                thinking_label="Reviewing diff",
                header="Review",
            )
            console.rule()
            continue

        if stripped == "/commit" or stripped.startswith("/commit "):
            msg = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
            if not msg:
                diff_text = tools.git_diff()
                if not diff_text.strip():
                    diff_text = tools.git_diff(staged=True)
                if not diff_text.strip():
                    console.print("[yellow]Nothing to commit.[/yellow]")
                    continue
                with Spinner("Generating commit message"):
                    msg = agent.ask(
                        "Write a single concise git commit message (conventional-commits style, "
                        "one subject line, optional short body) for this diff. Output only the "
                        "message, no code fences.\n\n```diff\n" + diff_text[:12000] + "\n```",
                    ).strip().strip("`").strip()
                console.print(f"[dim]Proposed:[/dim] {msg.splitlines()[0] if msg else '(empty)'}")
                if Prompt.ask("Commit with this message?", choices=["y", "n"], default="y") != "y":
                    console.print("[yellow]Commit cancelled.[/yellow]")
                    continue
            result = tools.git_commit(msg, add_all=True)
            console.print(f"[green]{result}[/green]")
            continue

        if stripped == "/doctor":
            import requests as _rq
            base = getattr(agent.model, "base_url", "http://localhost:11434")
            console.rule("Doctor")
            console.print(f"Configured model: [bold]{agent.model.model_name}[/bold]")
            try:
                resp = _rq.get(f"{base}/api/tags", timeout=10)
                resp.raise_for_status()
                installed = [m.get("name", "?") for m in (resp.json().get("models") or [])]
                console.print(f"[green]Ollama reachable[/green] at {base} ({len(installed)} models).")
                present = agent.model.model_name in installed
                mark = "[green]yes[/green]" if present else "[yellow]no (will fall back)[/yellow]"
                console.print(f"Configured model installed: {mark}")
                if installed:
                    console.print("[dim]Installed: " + ", ".join(installed[:20]) + "[/dim]")
            except Exception as exc:
                console.print(f"[red]Ollama not reachable[/red] at {base}: {exc}")
            console.print(f"MCP servers: {len(agent.mcp.status()) if agent.mcp else 0}")
            console.rule()
            continue

        if stripped == "/export" or stripped.startswith("/export "):
            path = stripped.split(maxsplit=1)[1].strip() if " " in stripped else "conversation.md"
            lines = ["# Conversation export\n"]
            for m in agent.history:
                role = m.get("role", "?")
                content = m.get("content") or ""
                if not isinstance(content, str):
                    import json as _json
                    content = _json.dumps(content, ensure_ascii=False)
                lines.append(f"\n## {role}\n\n{content}\n")
            try:
                tools.write_file(path, "".join(lines))
                console.print(f"[green]Exported {len(agent.history)} messages to[/green] {path}")
            except Exception as exc:
                console.print(f"[red]Export failed:[/red] {exc}")
            continue

        if stripped == "/export-sft" or stripped.startswith("/export-sft "):
            out_arg = stripped.split(maxsplit=1)[1].strip() if " " in stripped else "sft_dataset.jsonl"
            out_path = cfg.project_root / out_arg
            from sft_export import export_sft_dataset
            from sessions import DEFAULT_SESSIONS_DIR
            stats = export_sft_dataset(cfg.project_root / DEFAULT_SESSIONS_DIR, out_path)
            if stats["sessions_exported"] == 0:
                console.print(
                    f"[yellow]Nothing to export[/yellow] ({stats['sessions_seen']} session(s) "
                    "scanned, none had a clean user/assistant exchange)."
                )
            else:
                console.print(
                    f"[green]Exported {stats['pairs_exported']} exchange(s) from "
                    f"{stats['sessions_exported']}/{stats['sessions_seen']} session(s)[/green] "
                    f"to {out_path} (JSONL, one chat-SFT example per line; tool calls excluded)."
                )
            continue

        if stripped.startswith("/rename "):
            current_session_title = stripped.split(maxsplit=1)[1].strip()[:120]
            save_active_session()
            console.print(f"[green]Conversation renamed:[/green] {current_session_title}")
            continue

        if stripped.startswith("/delete-session "):
            target_id = stripped.split(maxsplit=1)[1].strip()
            if session_store.delete(target_id):
                console.print(f"[green]Deleted session:[/green] {target_id}")
                if target_id == current_session_id:
                    current_session_id = uuid4().hex[:12]
                    current_session_title = "New chat"
                    agent.history = agent.history[:1]
                    agent.tasks = []
                    agent.plan_mode = False
                    save_active_session()
            else:
                console.print(f"[yellow]No saved session named {target_id}.[/yellow]")
            continue

        if stripped == "/tokens" or stripped == "/cost":
            console.print(
                f"[bold]Estimated tokens this session[/bold] (chars/4, no tokenizer shipped):\n"
                f"  in:  {agent.total_tokens_in:,}\n"
                f"  out: {agent.total_tokens_out:,}\n"
                f"  total: {agent.total_tokens_in + agent.total_tokens_out:,}"
            )
            continue

        if stripped.startswith("/upload ") or stripped.startswith("/attach "):
            raw = stripped.split(maxsplit=1)[1]
            raw_path, prompt = _split_path_and_prompt(raw)
            up_path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
            if not up_path.is_file():
                console.print(f"[red]File not found:[/red] {up_path}")
                continue
            console.print(f"[dim]Uploading {up_path.name}...[/dim]")
            with Spinner(f"Reading {up_path.name}"):
                block = _ingest_upload(up_path, model_client)
            # Inject the file into the conversation so the agent can see it.
            agent.history.append({
                "role": "user",
                "content": f"[Uploaded file: {up_path.name}]\n{block}",
            })
            console.print(f"[green]Uploaded[/green] {up_path.name} "
                          f"({len(block):,} chars) into the conversation.")
            if prompt:
                _ask_streaming(
                    agent, prompt, ui_state,
                    on_tool_call=_on_tool_call, on_approval=_approval_prompt,
                )
                save_active_session()
            else:
                console.print("[dim]Ask me anything about it now.[/dim]")
            continue

        if stripped == "/paste" or stripped.startswith("/paste "):
            prompt = stripped.split(maxsplit=1)[1].strip() if " " in stripped else ""
            with Spinner("Reading clipboard"):
                pasted = _paste_from_clipboard(model_client)
            if pasted is None:
                console.print("[yellow]Clipboard is empty (or holds an image but Pillow "
                              "isn't installed: pip install pillow).[/yellow]")
                continue
            label, block = pasted
            agent.history.append({
                "role": "user",
                "content": f"[Pasted from clipboard: {label}]\n{block}",
            })
            console.print(f"[green]Pasted {label}[/green] ({len(block):,} chars) into the conversation.")
            if prompt:
                _ask_streaming(
                    agent, prompt, ui_state,
                    on_tool_call=_on_tool_call, on_approval=_approval_prompt,
                )
                save_active_session()
            else:
                console.print("[dim]Ask me anything about it now.[/dim]")
            continue

        if stripped.startswith("/agent "):
            task = stripped.split(maxsplit=1)[1]
            console.print("[dim]Spawning subagent...[/dim]")
            result = agent.spawn_subagent(task)
            console.rule("Subagent result")
            console.print(result)
            console.rule()
            continue

        if stripped.startswith("/loop "):
            rest = stripped.split(maxsplit=1)[1].strip()
            # Optional leading "cycles=N" or "every=S" tokens, then the goal.
            cycles = cfg.loop_max_cycles
            interval = cfg.loop_interval_seconds
            tokens = rest.split()
            while tokens and ("=" in tokens[0]):
                key, _, val = tokens[0].partition("=")
                if key in {"cycles", "n"} and val.isdigit():
                    cycles = int(val)
                elif key in {"every", "interval"}:
                    try:
                        interval = float(val)
                    except ValueError:
                        pass
                else:
                    break
                tokens = tokens[1:]
            goal = " ".join(tokens).strip()
            if not goal:
                console.print("[yellow]Usage: /loop [cycles=N] [every=SECONDS] <goal>[/yellow]")
                continue

            def on_cycle(n: int, reply: str) -> None:
                console.rule(f"Loop cycle {n}")
                console.print(reply)

            console.print(
                f"[green]Starting goal loop[/green] "
                f"(max {cycles} cycles, {interval}s interval). Press Ctrl-C to stop.\n"
                f"[dim]Goal:[/dim] {goal}"
            )
            try:
                summary = agent.run_goal_loop(
                    goal,
                    max_cycles=cycles,
                    interval=interval,
                    on_tool_call=_on_tool_call,
                    on_cycle=on_cycle,
                )
                console.print(f"[bold green]{summary}[/bold green]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Loop interrupted by user.[/yellow]")
            continue

        if stripped.split(maxsplit=1)[0] in {"/refine", "/best"}:
            command, _, rest = stripped.partition(" ")
            option = "rounds" if command == "/refine" else "n"
            default_count = 1 if command == "/refine" else 3
            count = default_count
            rest = rest.strip()
            if rest.startswith(f"{option}="):
                head, _, rest = rest.partition(" ")
                try:
                    count = max(1, min(int(head.split("=", 1)[1]), 8))
                except ValueError:
                    console.print(f"[yellow]{option}= needs a whole number.[/yellow]")
                    continue
            if not rest.strip():
                console.print(f"[yellow]Usage: {command} \\[{option}=N] <prompt>[/yellow]")
                continue
            label = "Refining" if command == "/refine" else f"Sampling {count} answers"
            try:
                with Spinner(label):
                    if command == "/refine":
                        reply = agent.ask_with_reflection(
                            rest, rounds=count,
                            on_tool_call=_on_tool_call, on_approval=_approval_prompt,
                        )
                    else:
                        reply = agent.ask_best_of_n(rest, n=count)
            except KeyboardInterrupt:
                agent.recover_from_interruption("")
                console.print("\n[yellow]Interrupted.[/yellow]")
                continue
            console.print("[bold magenta]Agent[/bold magenta]: ", end="")
            console.print(reply, markup=False, highlight=False)
            if command == "/refine":
                if agent.last_critiques:
                    console.print(f"[dim]Revised after {len(agent.last_critiques)} critique(s).[/dim]")
                else:
                    console.print("[dim]Reviewer approved the first draft.[/dim]")
            else:
                console.print(f"[dim]Picked the best of {len(agent.last_candidates)} candidates.[/dim]")
            save_active_session()
            continue

        if stripped in {"/image", "/images"}:
            console.print("[yellow]Usage: /image <path> [prompt][/yellow]")
            continue
        if stripped.startswith("/images "):
            stripped = "/image" + stripped[len("/images"):]
        if stripped.startswith("/image "):
            parts = stripped.split(maxsplit=2)
            img_path = parts[1]
            prompt = parts[2] if len(parts) > 2 else "Describe this image in detail."
            try:
                console.print(f"[dim]Analyzing image: {img_path}[/dim]")
                b64_data, mime = encode_image_base64(img_path)
                desc = model_client.describe_image(
                    image_base64=b64_data, prompt=prompt, mime_type=mime
                )
                meta = get_image_info(img_path)
                console.rule(f"Image: {img_path}")
                console.print(meta)
                console.print(f"\n[bold]AI Description:[/bold]\n{desc}")
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error analyzing image:[/red] {exc}")
            continue

        if stripped.startswith("/data "):
            parts = stripped.split(maxsplit=2)
            data_path = parts[1]
            chart_path = parts[2] if len(parts) > 2 else None
            result = analyze_dataset(
                tools._resolve_path(data_path),
                chart_path=tools._resolve_path(chart_path) if chart_path else None,
            )
            console.rule(f"Data: {data_path}")
            console.print(result)
            console.rule()
            continue

        if stripped.startswith("/audio "):
            audio_path = stripped.split(maxsplit=1)[1]
            try:
                console.print(f"[dim]Transcribing audio: {audio_path}[/dim]")
                transcript = transcribe_audio(audio_path)
                console.rule(f"Audio: {audio_path}")
                console.print(transcript)
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error transcribing audio:[/red] {exc}")
            continue

        if stripped in {"/video", "/videos"}:
            console.print("[yellow]Usage: /video <path> [prompt][/yellow]")
            continue
        if stripped.startswith("/videos "):
            stripped = "/video" + stripped[len("/videos"):]
        if stripped.startswith("/video "):
            parts = stripped.split(maxsplit=2)
            vid_path = parts[1]
            prompt = parts[2] if len(parts) > 2 else "Describe what is happening in this video."
            try:
                console.print(f"[dim]Analyzing video frames & audio: {vid_path}[/dim]")
                result = analyze_video(vid_path)
                console.rule(f"Video: {vid_path}")
                console.print(result["metadata"])
                console.print(f"[green]Extracted {result['frame_count']} key frames.[/green]")
                if result["frames"]:
                    try:
                        frame_desc = model_client.describe_images(result["frames"], prompt=prompt)
                        console.print(f"\n[bold]Visual Analysis:[/bold]\n{frame_desc}")
                    except Exception:
                        pass
                if result["transcript"]:
                    console.print(f"\n[bold]Audio Transcript:[/bold]\n{result['transcript']}")
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error analyzing video:[/red] {exc}")
            continue

        if stripped.startswith("/media "):
            med_path = stripped.split(maxsplit=1)[1]
            try:
                console.print(f"[dim]Analyzing media: {med_path}[/dim]")
                res = analyze_media(med_path)
                console.rule(f"Media: {med_path}")
                console.print(res)
                console.rule()
            except Exception as exc:
                console.print(f"[red]Error analyzing media:[/red] {exc}")
            continue

        # Custom slash commands loaded from .commands/*.md
        if stripped.startswith("/"):
            cmd_name = stripped[1:].split(maxsplit=1)[0].lower()
            if cmd_name in custom_commands:
                cmd_args = stripped[1:].split(maxsplit=1)[1] if " " in stripped else ""
                template = custom_commands[cmd_name]
                message_to_send = template.replace("$ARGUMENTS", cmd_args)
                console.print(f"[dim]Running custom command /{cmd_name}...[/dim]")
            else:
                console.print(f"[yellow]Unknown command: {stripped.split()[0]}. Try /help.[/yellow]")
                continue
        else:
            # Expand @file mentions into inline file contents.
            message_to_send = _expand_file_mentions(user_input, cfg.project_root)
        
        # Autonomous features
        if stripped.startswith("/continuous "):
            task = stripped.split(maxsplit=1)[1]
            console.print("[bold cyan]Starting continuous coding...[/bold cyan]")
            
            def on_progress(msg):
                console.print(f"[dim]{msg}[/dim]")
            
            result = autonomous.continuous_code(task, on_progress=on_progress)
            console.print(f"\n[bold green]Continuous coding complete![/bold green]")
            console.print(result)
            continue
        
        if stripped.startswith("/iterate "):
            task = stripped.split(maxsplit=1)[1]
            console.print("[bold cyan]Starting iterative generation...[/bold cyan]")
            
            def on_iteration(iteration, result, score):
                console.print(f"[dim]Iteration {iteration}: quality score {score:.2f}[/dim]")
            
            result = autonomous.iterative_generate(task, on_iteration=on_iteration)
            console.print(f"\n[bold green]Iterative generation complete![/bold green]")
            console.print(result)
            continue
        
        if stripped.startswith("/autonomous "):
            task = stripped.split(maxsplit=1)[1]
            console.print("[bold cyan]Starting autonomous workflow with task decomposition...[/bold cyan]")
            
            def on_task_complete(task_obj):
                console.print(f"[green]✓[/green] Completed: {task_obj.description}")
            
            result = autonomous.decompose_and_execute(task, on_task_complete=on_task_complete)
            console.print(f"\n[bold green]Autonomous workflow complete![/bold green]")
            console.print(result["summary"])
            continue
        
        if stripped.startswith("/stream "):
            task = stripped.split(maxsplit=1)[1]
            console.print("[bold cyan]Starting streaming generation...[/bold cyan]\n")
            
            def on_chunk(chunk):
                console.print(chunk, end="", markup=False)
            
            result = autonomous.stream_generate(task, on_chunk=on_chunk)
            console.print("\n\n[bold green]Streaming complete![/bold green]")
            continue
        
        if stripped.startswith("/autocorrect "):
            file_path = stripped.split(maxsplit=1)[1]
            try:
                content = tools.read_file(file_path)
                console.print(f"[dim]Auto-correcting {file_path}...[/dim]")
                
                corrected = autonomous.auto_correct(content)
                tools.write_file(file_path, corrected)
                
                console.print(f"[green]✓ Auto-corrected and saved {file_path}[/green]")
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
            continue

        # Normal conversation — agent uses tools automatically
        skill_names = agent.preview_skill_names(message_to_send)
        if skill_names:
            console.print(f"[dim]Using skills: {', '.join(skill_names)}[/dim]")
        mode_flags = []
        if agent.plan_mode:
            mode_flags.append("plan")
        if agent.require_approval:
            mode_flags.append("approval")
        flag_str = f" [{'+'.join(mode_flags)}]" if mode_flags else ""
        _ask_streaming(
            agent,
            message_to_send,
            ui_state,
            on_tool_call=_on_tool_call,
            on_approval=_approval_prompt,
            thinking_label=f"Thinking in {agent.mode} mode{flag_str}",
        )
        save_active_session()

        # After each agent reply, show a brief changes summary if any new changes occurred
        if tools.changes.entries:
            recent = tools.changes.recent(5)
            if recent:
                console.print("[dim]Recent changes:[/dim]")
                for entry in recent:
                    import time as _t
                    t = _t.strftime("%H:%M:%S", _t.localtime(entry.timestamp))
                    color_map = {
                        "create": "green", "write": "yellow", "patch": "blue",
                        "delete_file": "red", "delete_code": "red", "mkdir": "cyan",
                    }
                    color = color_map.get(entry.action, "white")
                    details = f"  {entry.details}" if entry.details else ""
                    console.print(
                        f"  [dim]{t}[/dim] [{color}]{entry.action}[/{color}] "
                        f"[dim]{entry.path}{details}[/dim]"
                    )
                console.print("[dim]Use /changes to see the full log.[/dim]")

    if agent.mcp:
        agent.mcp.close()


if __name__ == "__main__":
    app()
