from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, ClassVar, Optional

from urllib.parse import urlparse

import shutil
import requests


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "node_modules",
}

# Change tracking
@dataclass
class ChangeEntry:
    """A single recorded change made by the agent."""
    timestamp: float
    action: str        # "create", "write", "patch", "delete_file", "delete_code", "mkdir"
    path: str
    details: str = ""
    old_snippet: str = ""
    new_snippet: str = ""
    pre_text: str = ""
    post_text: str = ""
    old_start_line: int = 1

    def summary(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        suffix = f"  — {self.details}" if self.details else ""
        return f"[{t}] {self.action.upper():12s} {self.path}{suffix}"


@dataclass
class ChangeTracker:
    """Records all file changes made during a session."""
    entries: List[ChangeEntry] = field(default_factory=list)

    def record(
        self,
        action: str,
        path: str,
        details: str = "",
        old_snippet: str = "",
        new_snippet: str = "",
        pre_text: str = "",
        post_text: str = "",
        old_start_line: int = 1,
    ) -> ChangeEntry:
        entry = ChangeEntry(
            timestamp=time.time(),
            action=action,
            path=path,
            details=details,
            old_snippet=old_snippet,
            new_snippet=new_snippet,
            pre_text=pre_text,
            post_text=post_text,
            old_start_line=old_start_line,
        )
        self.entries.append(entry)
        return entry

    def summary(self) -> str:
        if not self.entries:
            return "No changes recorded yet."
        lines = [f"Total changes: {len(self.entries)}"]
        for entry in self.entries:
            lines.append(entry.summary())
        return "\n".join(lines)

    def clear(self) -> None:
        self.entries.clear()

    def recent(self, n: int = 10) -> List[ChangeEntry]:
        return self.entries[-n:]

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.entries:
            counts[e.action] = counts.get(e.action, 0) + 1
        return counts

# Project tools
@dataclass
class ProjectTools:
    """
    Utilities for safely interacting with files inside a project root.
    """
    active_instance: ClassVar[Optional[ProjectTools]] = None

    project_root: Path
    command_timeout: int = 3600
    include_ignored_files: bool = True
    restrict_to_project: bool = False
    changes: ChangeTracker = field(default_factory=ChangeTracker)

    def __post_init__(self) -> None:
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root).resolve()
        ProjectTools.active_instance = self

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        """
        Resolve the given path. If restrict_to_project is True, ensure it stays within project_root.
        Otherwise allows full unrestricted path access.
        """
        p = Path(relative_or_absolute)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.expanduser().resolve()
        if self.restrict_to_project:
            try:
                p.relative_to(self.project_root.resolve())
            except ValueError:
                raise ValueError("Refusing to access a path outside the project root.")
        return p

    def list_files(self, subdir: str | None = None, pattern: str | None = None) -> List[Path]:
        base = self.project_root if subdir is None else self._resolve_path(subdir)
        if not base.exists():
            return []
        paths: List[Path] = []
        iterator = base.rglob(pattern or "*")
        for p in iterator:
            if not p.is_file():
                continue
            relative_parts = p.relative_to(base).parts
            if (
                not self.include_ignored_files
                and any(part in DEFAULT_IGNORED_DIRS for part in relative_parts)
            ):
                continue
            paths.append(p)
        return paths

    def read_file(self, path: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str, backup: bool = True) -> None:
        target = self._resolve_path(path)
        is_new = not target.exists()
        old_text = ""
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup and target.exists():
            backup_path = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup_path)
            try:
                old_text = backup_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                old_text = ""
        target.write_text(content, encoding="utf-8")
        action = "create" if is_new else "write"
        line_count = len(content.splitlines())
        self.changes.record(
            action,
            path,
            f"{line_count} lines",
            pre_text=old_text,
            post_text=content,
        )

    def apply_patch(self, path: str, old_snippet: str, new_snippet: str, backup: bool = True) -> bool:
        """
        Replace the first occurrence of old_snippet with new_snippet in the file.
        Returns True if a replacement was made, False otherwise.
        """
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        text = target.read_text(encoding="utf-8")
        if old_snippet not in text:
            return False
        if backup:
            backup_path = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup_path)
        start_char = text.find(old_snippet)
        old_start_line = text[:start_char].count('\n') + 1 if start_char != -1 else 1
        updated = text.replace(old_snippet, new_snippet, 1)
        target.write_text(updated, encoding="utf-8")
        old_lines = len(old_snippet.splitlines())
        new_lines = len(new_snippet.splitlines())
        self.changes.record(
            "patch",
            path,
            f"-{old_lines} +{new_lines} lines",
            old_snippet=old_snippet,
            new_snippet=new_snippet,
            pre_text=text,
            post_text=updated,
            old_start_line=old_start_line,
        )
        return True

    def delete_file(self, path: str) -> bool:
        """
        Delete a file from the project. Returns True if deleted.
        """
        target = self._resolve_path(path)
        if not target.exists():
            return False
        old_text = ""
        try:
            old_text = target.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        target.unlink()
        self.changes.record("delete_file", path, pre_text=old_text)
        return True

    def delete_code(self, path: str, snippet: str, backup: bool = True) -> bool:
        """
        Remove the first occurrence of a code snippet from a file.
        Returns True if the snippet was found and removed.
        """
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        text = target.read_text(encoding="utf-8")
        if snippet not in text:
            return False
        if backup:
            backup_path = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup_path)
        start_char = text.find(snippet)
        old_start_line = text[:start_char].count('\n') + 1 if start_char != -1 else 1
        updated = text.replace(snippet, "", 1)
        target.write_text(updated, encoding="utf-8")
        removed_lines = len(snippet.splitlines())
        self.changes.record(
            "delete_code",
            path,
            f"-{removed_lines} lines removed",
            old_snippet=snippet,
            new_snippet="",
            pre_text=text,
            post_text=updated,
            old_start_line=old_start_line,
        )
        return True

    def add_file(self, path: str) -> str:
        """
        Create an empty file at the given path (creates parent dirs as needed).
        If the file already exists it is overwritten with empty content.
        Returns the resolved absolute path as a string.
        """
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        self.changes.record("create", path, "0 lines (empty file)")
        return str(target)

    def create_directory(self, path: str) -> str:
        """
        Create a directory (and parents) inside the project.
        Returns the resolved path as a string.
        """
        target = self._resolve_path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.changes.record("mkdir", path)
        return str(target)

    def read_lines(self, path: str, start_line: int = 1, end_line: int = 100) -> str:
        """
        Read a slice of a file with line numbers (1-indexed), like Claude Code.
        """
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        start = max(1, start_line)
        end = min(total_lines, end_line)
        if start > total_lines:
            return f"File has {total_lines} lines (requested start {start_line} is beyond EOF)."
        formatted = []
        for i in range(start, end + 1):
            formatted.append(f"{i:5d}: {lines[i - 1]}")
        header = f"[{target.name} lines {start}-{end} of {total_lines}]\n"
        return header + "\n".join(formatted)

    def grep_search(self, query: str, subdir: str | None = None, is_regex: bool = False, max_matches: int = 50) -> str:
        """
        Search for text or regex across files in the project. Returns matching files, line numbers, and snippets.
        """
        import re
        base = self.project_root if subdir is None else self._resolve_path(subdir)
        if not base.exists():
            return f"Directory not found: {base}"
        
        flags = re.IGNORECASE
        pattern = re.compile(query if is_regex else re.escape(query), flags)
        matches = []
        
        for file_path in base.rglob("*"):
            if not file_path.is_file():
                continue
            relative_parts = file_path.relative_to(self.project_root).parts
            if not self.include_ignored_files and any(p in DEFAULT_IGNORED_DIRS for p in relative_parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for line_no, line in enumerate(content.splitlines(), start=1):
                    if pattern.search(line):
                        rel_path = str(file_path.relative_to(self.project_root))
                        snippet = line.strip()[:150]
                        matches.append(f"{rel_path}:{line_no}: {snippet}")
                        if len(matches) >= max_matches:
                            return f"Found {len(matches)}+ matches (capped at {max_matches}):\n" + "\n".join(matches)
            except Exception:
                continue

        if not matches:
            return f"No matches found for '{query}'."
        return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

    def get_outline(self, path: str) -> str:
        """
        Extract symbols (classes, functions, methods, imports) from a code file using AST.
        """
        import ast
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        
        if target.suffix == ".py":
            try:
                tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"), filename=str(target))
                symbols = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        symbols.append(f"Class: {node.name} (line {node.lineno})")
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                symbols.append(f"  Method: {item.name}() (line {item.lineno})")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(f"Function: {node.name}() (line {node.lineno})")
                    elif isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                        symbols.append(f"Import: {', '.join(names)} (line {node.lineno})")
                    elif isinstance(node, ast.ImportFrom):
                        names = [alias.name for alias in node.names]
                        symbols.append(f"From {node.module or ''} import {', '.join(names)} (line {node.lineno})")
                if not symbols:
                    return f"No top-level functions or classes found in {target.name}."
                return f"Outline of {target.name}:\n" + "\n".join(symbols)
            except SyntaxError as e:
                return f"Syntax error in {target.name} at line {e.lineno}: {e.msg}"
        
        # Generic outline using regex for non-python files
        import re
        content = target.read_text(encoding="utf-8", errors="replace")
        func_pattern = re.compile(r'^\s*(?:async\s+)?(?:def|function|class|interface|type)\s+([A-Za-z0-9_]+)', re.MULTILINE)
        found = [f"{m.group(0).strip()} (line {content[:m.start()].count(chr(10)) + 1})" for m in func_pattern.finditer(content)]
        if found:
            return f"Outline of {target.name}:\n" + "\n".join(found[:50])
        return f"File {target.name}: {len(content.splitlines())} lines."

    def undo_last_change(self, path: str | None = None) -> str:
        """
        Revert the most recent file change using its .bak backup.
        """
        if path:
            target = self._resolve_path(path)
            bak = target.with_suffix(target.suffix + ".bak")
            if not bak.exists():
                return f"No backup file (.bak) found for {path}."
            shutil.copy2(bak, target)
            bak.unlink()
            self.changes.record("undo", path, "Restored from backup")
            return f"Successfully restored {path} from backup."
        
        # If no path specified, look at recent changes
        for entry in reversed(self.changes.entries):
            if entry.action in {"write", "patch", "delete_code"}:
                target = self._resolve_path(entry.path)
                bak = target.with_suffix(target.suffix + ".bak")
                if bak.exists():
                    shutil.copy2(bak, target)
                    bak.unlink()
                    self.changes.record("undo", entry.path, "Restored from backup")
                    return f"Successfully restored {entry.path} to previous version."
        return "No recent changes with available backup (.bak) found to undo."

    def git_status(self) -> str:
        """Get git repository status."""
        return self.run_command("git status --short -b")

    def git_diff(self, file_path: str | None = None, staged: bool = False) -> str:
        """Get git diff for the repo or a specific file."""
        cmd = "git diff"
        if staged:
            cmd += " --cached"
        if file_path:
            cmd += f" -- {file_path}"
        output = self.run_command(cmd)
        return output if output else "No git diff."

    def git_commit(self, message: str, add_all: bool = True) -> str:
        """Stage files and create a git commit."""
        if add_all:
            add_res = self.run_command("git add -A")
            if "Error" in add_res:
                return f"Git add failed: {add_res}"
        res = self.run_command(f'git commit -m "{message}"')
        return res

    def validate_syntax(self, path: str) -> str:
        """
        Validate syntax of a Python, JSON, or YAML file before saving or running.
        """
        target = self._resolve_path(path)
        if not target.exists():
            return f"File not found: {target}"
        content = target.read_text(encoding="utf-8", errors="replace")
        ext = target.suffix.lower()
        if ext == ".py":
            import ast
            try:
                ast.parse(content, filename=str(target))
                return f"OK: {target.name} has valid Python syntax."
            except SyntaxError as e:
                return f"Syntax Error in {target.name} at line {e.lineno}, col {e.offset}: {e.msg}\n  {e.text}"
        elif ext == ".json":
            import json
            try:
                json.loads(content)
                return f"OK: {target.name} has valid JSON syntax."
            except Exception as e:
                return f"JSON Parse Error in {target.name}: {e}"
        elif ext in {".yaml", ".yml"}:
            try:
                import yaml
                yaml.safe_load(content)
                return f"OK: {target.name} has valid YAML syntax."
            except Exception as e:
                return f"YAML Parse Error in {target.name}: {e}"
        return f"OK: {target.name} format checked."

    def run_command(self, command: str, timeout: int | None = None) -> str:
        """
        Run a shell command in the project directory and return stdout/stderr.
        """
        import subprocess

        effective_timeout = timeout or self.command_timeout

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(self.project_root),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            output += f"\nExit code: {result.returncode}"
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"Command timed out after {effective_timeout} seconds."
        except Exception as e:
            return f"Error running command: {e}"


def fetch_url(url: str, timeout: int = 15) -> str:
    """
    Fetch a URL over HTTP/HTTPS and return its text content, converting HTML to Markdown for readability.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify as md
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove noisy elements
            for element in soup(["script", "style", "nav", "footer", "aside"]):
                element.extract()
            text = md(str(soup), heading_style="ATX").strip()
            # Collapse multiple newlines
            import re
            return re.sub(r'\n{3,}', '\n\n', text)
        except ImportError:
            pass
    return resp.text

def http_request(
    method: str,
    url: str,
    headers: Dict[str, str] | None = None,
    params: Dict[str, str] | None = None,
    json_body: Dict | list | None = None,
    data: str | None = None,
    timeout: int = 30,
    max_chars: int = 20000,
) -> str:
    """
    Make a general HTTP/HTTPS request to an external API and return a compact
    summary of the response (status, key headers, and body).

    Supports any method (GET/POST/PUT/PATCH/DELETE), custom headers, query
    params, and either a JSON or a raw text body. The body is truncated to
    ``max_chars`` so large responses don't flood the context.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")

    method = (method or "GET").upper()
    allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
    if method not in allowed:
        raise ValueError(f"Unsupported HTTP method: {method}")

    req_headers = {"User-Agent": "my-ai-agent/1.0"}
    if headers:
        req_headers.update(headers)

    resp = requests.request(
        method,
        url,
        headers=req_headers,
        params=params or None,
        json=json_body if json_body is not None else None,
        data=data if data is not None else None,
        timeout=timeout,
    )

    body = resp.text or ""
    truncated = ""
    if len(body) > max_chars:
        truncated = f"\n...[truncated, {len(body) - max_chars} more chars]"
        body = body[:max_chars]

    content_type = resp.headers.get("Content-Type", "")
    summary = (
        f"{method} {url}\n"
        f"Status: {resp.status_code} {resp.reason}\n"
        f"Content-Type: {content_type}\n"
        f"--- Body ---\n{body}{truncated}"
    )
    return summary


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        engine = DDGS()
        results = list(engine.text(query, max_results=max_results))
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        if not results:
            return "No results found."
        formatted = []
        for i, res in enumerate(results, 1):
            formatted.append(f"{i}. {res.get('title')}\n   URL: {res.get('href')}\n   Snippet: {res.get('body')}")
        return "\n\n".join(formatted)
    except ImportError:
        return "Error: web search dependency is missing. Install the project requirements and retry."
    except Exception as e:
        return f"Search failed: {e}"
