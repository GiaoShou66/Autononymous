from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

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

    def summary(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        suffix = f"  — {self.details}" if self.details else ""
        return f"[{t}] {self.action.upper():12s} {self.path}{suffix}"


@dataclass
class ChangeTracker:
    """Records all file changes made during a session."""
    entries: List[ChangeEntry] = field(default_factory=list)

    def record(self, action: str, path: str, details: str = "") -> None:
        self.entries.append(ChangeEntry(
            timestamp=time.time(),
            action=action,
            path=path,
            details=details,
        ))

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

    project_root: Path
    command_timeout: int = 300
    include_ignored_files: bool = False
    changes: ChangeTracker = field(default_factory=ChangeTracker)

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        """
        Resolve the given path and ensure it stays within project_root.
        """
        p = Path(relative_or_absolute)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.expanduser().resolve()
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
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup and target.exists():
            backup_path = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup_path)
        target.write_text(content, encoding="utf-8")
        action = "create" if is_new else "write"
        line_count = len(content.splitlines())
        self.changes.record(action, path, f"{line_count} lines")

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
        updated = text.replace(old_snippet, new_snippet, 1)
        target.write_text(updated, encoding="utf-8")
        old_lines = len(old_snippet.splitlines())
        new_lines = len(new_snippet.splitlines())
        self.changes.record("patch", path, f"-{old_lines} +{new_lines} lines")
        return True

    def delete_file(self, path: str) -> bool:
        """
        Delete a file from the project. Returns True if deleted.
        """
        target = self._resolve_path(path)
        if not target.exists():
            return False
        target.unlink()
        self.changes.record("delete_file", path)
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
        updated = text.replace(snippet, "", 1)
        target.write_text(updated, encoding="utf-8")
        removed_lines = len(snippet.splitlines())
        self.changes.record("delete_code", path, f"-{removed_lines} lines removed")
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

def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            formatted = []
            for i, res in enumerate(results, 1):
                formatted.append(f"{i}. {res.get('title')}\n   URL: {res.get('href')}\n   Snippet: {res.get('body')}")
            return "\n\n".join(formatted)
    except ImportError:
        return "Error: duckduckgo-search is not installed. Please run `pip install duckduckgo-search`."
    except Exception as e:
        return f"Search failed: {e}"
