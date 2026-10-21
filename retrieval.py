"""Hierarchical retrieval and summarization over the project.

Retrieval is two-level, coarse to fine:
  1. directories are scored against the query using the vocabulary of the
     files they directly contain (plus their own path),
  2. only files inside the best directories are ranked, and each hit carries
     a one-line summary plus the symbols found in it with line numbers.

No embeddings or network calls: summaries come from the code itself (module
docstring, headings, top-level names) and ranking is IDF-weighted term
overlap. The index rebuilds itself when files change.

summarize_long() is the other hierarchy: map-reduce summarization of text too
long for one model call (chunks -> chunk summaries -> summaries of summaries).
"""

from __future__ import annotations

import ast
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".agent_sessions",
    ".impeccable", ".claude", "dist", "build", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".next", "target",
}
TEXT_EXTS = {
    ".py", ".md", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".html", ".css", ".sh", ".bat", ".toml", ".txt", ".rs", ".go", ".java",
    ".c", ".h", ".cpp",
}
MAX_FILE_BYTES = 300_000
MAX_SYMBOLS = 12
BODY_TERMS = 40

STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was be it this that from as at by "
    "not if else def return import self none true false new get set can will do how "
    "what where when which who why you your i we our they them".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_JS_SYMBOL = re.compile(
    r"(?:function|class)\s+([A-Za-z_]\w*)|const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\(|function)"
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, splitting snake_case and camelCase, minus stopwords."""
    tokens: List[str] = []
    for word in _WORD.findall(text):
        for part in _CAMEL.findall(word):
            part = part.lower()
            if len(part) >= 2 and part not in STOPWORDS:
                tokens.append(part)
    return tokens


@dataclass
class FileEntry:
    path: str
    summary: str
    terms: FrozenSet[str]


@dataclass
class Hit:
    path: str
    summary: str
    score: float


def _summarize_python(text: str) -> Optional[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    doc = ast.get_docstring(tree)
    head = doc.strip().splitlines()[0][:140] if doc else ""
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(f"{node.name} (L{node.lineno})")
        if len(symbols) >= MAX_SYMBOLS:
            break
    parts = [p for p in (head, ", ".join(symbols)) if p]
    return " | ".join(parts) if parts else None


def _summarize_markdown(text: str) -> Optional[str]:
    headings = [ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")]
    return ", ".join(headings[:8])[:240] if headings else None


def _summarize_scripty(text: str) -> Optional[str]:
    names = []
    for match in _JS_SYMBOL.finditer(text):
        names.append(match.group(1) or match.group(2))
        if len(names) >= MAX_SYMBOLS:
            break
    return ", ".join(names) if names else None


def summarize_file(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    summary = None
    if suffix == ".py":
        summary = _summarize_python(text)
    elif suffix == ".md":
        summary = _summarize_markdown(text)
    elif suffix in {".js", ".mjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".h", ".cpp"}:
        summary = _summarize_scripty(text)
    if summary:
        return summary
    first = [ln.strip() for ln in text.splitlines() if ln.strip()][:2]
    return " / ".join(ln[:100] for ln in first) or "(empty)"


class CodeIndex:
    """Two-level (directory -> file) searchable summary of a project tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.files: Dict[str, FileEntry] = {}
        self._signature: Optional[Tuple[int, int]] = None
        self._idf: Dict[str, float] = {}

    def _walk(self) -> List[Path]:
        found: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() not in TEXT_EXTS:
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                found.append(path)
        return found

    @staticmethod
    def _signature_of(paths: List[Path]) -> Tuple[int, int]:
        newest = 0
        for path in paths:
            try:
                newest = max(newest, path.stat().st_mtime_ns)
            except OSError:
                pass
        return len(paths), newest

    def ensure_fresh(self) -> bool:
        """Rebuild if files were added, removed, or modified. True if rebuilt."""
        paths = self._walk()
        signature = self._signature_of(paths)
        if signature == self._signature:
            return False
        self._build(paths)
        self._signature = signature
        return True

    def _build(self, paths: List[Path]) -> None:
        files: Dict[str, FileEntry] = {}
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix()
            summary = summarize_file(path, text)
            body_counts = Counter(tokenize(text))
            terms = set(tokenize(rel)) | set(tokenize(summary))
            terms |= {t for t, _ in body_counts.most_common(BODY_TERMS)}
            files[rel] = FileEntry(rel, summary, frozenset(terms))
        self.files = files
        doc_freq: Counter = Counter()
        for entry in files.values():
            doc_freq.update(entry.terms)
        total = max(1, len(files))
        self._idf = {t: math.log(1 + total / (1 + df)) for t, df in doc_freq.items()}

    @staticmethod
    def _directory(rel_path: str) -> str:
        parent = rel_path.rpartition("/")[0]
        return parent or "."

    def search(self, query: str, top_dirs: int = 3, top_files: int = 6) -> List[Hit]:
        self.ensure_fresh()
        query_terms = set(tokenize(query))
        if not query_terms or not self.files:
            return []

        dir_terms: Dict[str, set] = {}
        for rel, entry in self.files.items():
            dir_terms.setdefault(self._directory(rel), set()).update(entry.terms)

        def idf(term: str) -> float:
            return self._idf.get(term, 0.0)

        dir_scores = []
        for directory, terms in dir_terms.items():
            score = sum(idf(t) for t in query_terms & terms)
            score += 0.5 * sum(idf(t) for t in query_terms & set(tokenize(directory)))
            if score > 0:
                dir_scores.append((score, directory))
        dir_scores.sort(reverse=True)
        chosen = {d for _, d in dir_scores[:top_dirs]}

        hits: List[Hit] = []
        for rel, entry in self.files.items():
            if self._directory(rel) not in chosen:
                continue
            path_terms = set(tokenize(rel))
            score = 0.0
            for term in query_terms:
                if term in path_terms:
                    score += 3 * idf(term)
                elif term in entry.terms:
                    score += idf(term)
            if score > 0:
                hits.append(Hit(rel, entry.summary, round(score, 3)))
        hits.sort(key=lambda h: (-h.score, h.path))
        return hits[:top_files]

    @staticmethod
    def render(hits: List[Hit], max_chars: int = 3000) -> str:
        lines: List[str] = []
        used = 0
        for hit in hits:
            line = f"- {hit.path}: {hit.summary}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


def _chunks(text: str, size: int) -> List[str]:
    """Split on line boundaries into chunks of roughly `size` characters."""
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in text.splitlines(keepends=True):
        if current and length + len(line) > size:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def summarize_long(
    model: Any, text: str, chunk_chars: int = 6000, fanin: int = 4
) -> str:
    """Map-reduce summarization: chunk summaries, then summaries of groups of
    summaries, repeated until one remains. Short text takes a single call."""
    prompt = (
        "Summarize the following concisely, keeping concrete names, decisions, "
        "and anything a developer would need. Output only the summary.\n\n"
    )

    def ask(body: str) -> str:
        reply = model.chat([{"role": "user", "content": prompt + body}])
        return str(reply.get("content") or "").strip()

    pieces = _chunks(text, max(500, chunk_chars))
    if not pieces:
        return ""
    summaries = [ask(piece) for piece in pieces]
    fanin = max(2, fanin)
    while len(summaries) > 1:
        summaries = [
            ask("\n\n".join(summaries[i : i + fanin])) for i in range(0, len(summaries), fanin)
        ]
    return summaries[0]
