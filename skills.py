from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import re
import yaml


DEFAULT_SKILL_PROMPT_CHARS = 6000


COMMON_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "code",
    "create",
    "do",
    "file",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "make",
    "me",
    "my",
    "of",
    "on",
    "or",
    "project",
    "the",
    "this",
    "to",
    "use",
    "user",
    "want",
    "wants",
    "when",
    "with",
    "you",
}


FILE_EXTENSION_SKILLS = {
    "docx": ".docx",
    "pdf": ".pdf",
    "pptx": ".pptx",
    "xlsx": ".xlsx",
}


SKILL_ALIASES = {
    "docx": {"document", "documents", "doc", "word", "microsoft word"},
    "frontend-design": {
        "app",
        "css",
        "dashboard",
        "frontend",
        "html",
        "interface",
        "page",
        "react",
        "site",
        "ui",
        "web app",
        "website",
    },
    "pdf": {"acrobat", "form", "ocr", "pdfs", "report"},
    "pptx": {
        "deck",
        "keynote",
        "powerpoint",
        "presentation",
        "slide",
        "slides",
        "slideshow",
    },
    "webapp-testing": {"browser test", "e2e", "playwright", "test website"},
    "xlsx": {"excel", "sheet", "spreadsheet", "table", "workbook"},
}


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower()):
        if len(token) <= 2 or token in COMMON_WORDS:
            continue
        tokens.add(token)
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
    return tokens


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    instructions: str
    metadata: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    def format_for_prompt(self, max_chars: int = DEFAULT_SKILL_PROMPT_CHARS) -> str:
        instructions = self.instructions.strip()
        if len(instructions) > max_chars:
            instructions = (
                instructions[:max_chars].rstrip()
                + "\n\n...[skill instructions truncated; use read_skill_file for more]..."
            )

        return (
            f"## Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Skill files are rooted at: {self.root}\n\n"
            f"{instructions}"
        )


class SkillRegistry:
    """
    Loads Claude-style skills from folders containing SKILL.md.

    The registry is model-agnostic: it only discovers, scores, and safely reads
    skill resources. The agent decides when to inject the selected skills.

    New skills are hot-reloaded automatically: each call to ``get`` or ``match``
    checks for new or changed SKILL.md files on disk and reloads if needed.

    Scans exactly one root directory â the ``skills_root`` passed by the caller.
    """

    def __init__(self, skills_root: Path | str | None) -> None:
        self._skills: dict[str, Skill] = {}
        self._skill_mtimes: dict[Path, float] = {}
        self._roots: list[Path] = []

        if skills_root is not None:
            root = Path(skills_root).expanduser()
            if root.is_dir():
                self._roots.append(root)

        if self._roots:
            self._load_skills()

    @property
    def skills_root(self) -> Path | None:
        """Primary skills root (first one found, for backward compat)."""
        return self._roots[0] if self._roots else None

    @property
    def roots(self) -> list[Path]:
        """All active skill roots."""
        return list(self._roots)

    def reload(self) -> bool:
        """Rescan the skill root for new or changed SKILL.md files.

        Returns True if the registry was updated.
        """
        return self._load_skills()

    def _skill_files(self) -> dict[Path, float]:
        """Return a mapping of SKILL.md paths to their current mtimes.

        Only scans the single root directory passed to ``__init__``.
        """
        files: dict[Path, float] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("SKILL.md")):
                try:
                    resolved = path.resolve()
                    if resolved not in files:
                        files[resolved] = path.stat().st_mtime
                except OSError:
                    continue
        return files

    def _load_skills(self) -> bool:
        """Load or reload all skills. Returns True if the set changed."""
        current_files = self._skill_files()
        if current_files == self._skill_mtimes:
            return False

        skills: dict[str, Skill] = {}
        for skill_file in current_files:
            skill = self._parse_skill(skill_file)
            if skill.name:
                key = skill.name.lower()
                # First root wins on name collisions (earlier roots have
                # higher priority).
                if key not in skills:
                    skills[key] = skill

        self._skills = skills
        self._skill_mtimes = current_files
        return True

    def __bool__(self) -> bool:
        self.reload()
        return bool(self._skills)

    def __len__(self) -> int:
        self.reload()
        return len(self._skills)

    def all(self) -> List[Skill]:
        self.reload()
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        self.reload()
        return self._skills.get(name.strip().lower())

    def names(self) -> List[str]:
        return [skill.name for skill in self.all()]

    def match(self, query: str, limit: int = 3) -> List[Skill]:
        self.reload()
        query_text = query.lower()
        query_tokens = _tokens(query_text)
        if not query_tokens and not query_text.strip():
            return []

        scored: list[tuple[int, Skill]] = []
        for skill in self._skills.values():
            score = self._score_skill(skill, query_text, query_tokens)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for _, skill in scored[:limit]]

    def list_skill_files(self, skill_name: str) -> List[Path]:
        skill = self._require_skill(skill_name)
        return sorted(path for path in skill.root.rglob("*") if path.is_file())

    def read_skill_file(self, skill_name: str, relative_path: str) -> str:
        skill = self._require_skill(skill_name)
        base = skill.root.resolve()
        target = (base / relative_path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError("Refusing to access a file outside the skill folder.")
        if not target.is_file():
            raise FileNotFoundError(f"Skill file not found: {relative_path}")
        return target.read_text(encoding="utf-8")

    def _require_skill(self, skill_name: str) -> Skill:
        skill = self.get(skill_name)
        if not skill:
            raise KeyError(f"Unknown skill: {skill_name}")
        return skill

    def _score_skill(
        self,
        skill: Skill,
        query_text: str,
        query_tokens: set[str],
    ) -> int:
        skill_name = skill.name.lower()
        name_tokens = _tokens(skill_name.replace("-", " "))
        description_tokens = _tokens(skill.description)
        aliases = SKILL_ALIASES.get(skill_name, set())
        alias_tokens = _tokens(" ".join(aliases))

        score = 0
        if skill_name in query_text:
            score += 20

        extension = FILE_EXTENSION_SKILLS.get(skill_name)
        if extension and extension in query_text:
            score += 20

        for alias in aliases:
            if alias in query_text:
                score += 15

        score += 6 * len(query_tokens & name_tokens)
        score += 5 * len(query_tokens & alias_tokens)
        score += len(query_tokens & description_tokens)

        # A small boost for strong domain words inside the body, without making
        # the whole skill instruction text dominate matching.
        body_tokens = _tokens(skill.instructions[:2000])
        score += min(4, len(query_tokens & body_tokens))
        return score

    def _parse_skill(self, skill_file: Path) -> Skill:
        text = skill_file.read_text(encoding="utf-8")
        metadata, instructions = self._split_frontmatter(text)

        name = str(metadata.get("name") or skill_file.parent.name).strip()
        description = str(metadata.get("description") or "").strip()
        return Skill(
            name=name,
            description=description,
            path=skill_file.resolve(),
            instructions=instructions.strip(),
            metadata=metadata,
        )

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text

        end_index = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = idx
                break

        if end_index is None:
            return {}, text

        raw_metadata = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :])
        try:
            metadata = yaml.safe_load(raw_metadata) or {}
        except yaml.YAMLError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body
