from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import yaml


DEFAULT_MODEL_NAME = "llama3:latest"
DEFAULT_MAX_CONTEXT_TURNS = 50
DEFAULT_MAX_ITERATIONS = 60
DEFAULT_MAX_AUTO_SKILLS = 5
DEFAULT_SKILL_PROMPT_CHARS = 20000
DEFAULT_COMMAND_TIMEOUT = 300
DEFAULT_LIST_OUTPUT_LIMIT = 0
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_SKILLS_ROOT = Path(__file__).parent.resolve() / "skills-main" / "skills"


@dataclass
class Config:
    model_name: str = DEFAULT_MODEL_NAME
    max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_auto_skills: int = DEFAULT_MAX_AUTO_SKILLS
    skill_prompt_chars: int = DEFAULT_SKILL_PROMPT_CHARS
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    list_output_limit: int = DEFAULT_LIST_OUTPUT_LIMIT
    include_ignored_files: bool = False
    project_root: Path = Path.cwd()
    skills_root: Optional[Path] = DEFAULT_SKILLS_ROOT


def _load_from_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # If config loading fails, fall back to defaults.
        return {}


def _get_int(yaml_cfg: dict, key: str, env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None:
        value = yaml_cfg.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_bool(yaml_cfg: dict, key: str, env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        value = yaml_cfg.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    config_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Config:
    """
    Load configuration from (in order of precedence):
    - Explicit function arguments
    - Environment variables
    - YAML config file
    - Built-in defaults
    """
    if config_path is None:
        config_path = Path(DEFAULT_CONFIG_FILE)
        if not config_path.is_file():
            # Fallback to the directory where config.py is located
            fallback_path = Path(__file__).parent.resolve() / DEFAULT_CONFIG_FILE
            if fallback_path.is_file():
                config_path = fallback_path

    yaml_cfg = _load_from_yaml(config_path)

    model_name = (
        os.environ.get("AI_AGENT_MODEL", "").strip()
        or (yaml_cfg.get("model_name") or "").strip()
        or DEFAULT_MODEL_NAME
    )

    max_turns = _get_int(
        yaml_cfg,
        "max_context_turns",
        "AI_AGENT_MAX_TURNS",
        DEFAULT_MAX_CONTEXT_TURNS,
    )
    max_iterations = _get_int(
        yaml_cfg,
        "max_iterations",
        "AI_AGENT_MAX_ITERATIONS",
        DEFAULT_MAX_ITERATIONS,
    )
    max_auto_skills = _get_int(
        yaml_cfg,
        "max_auto_skills",
        "AI_AGENT_MAX_AUTO_SKILLS",
        DEFAULT_MAX_AUTO_SKILLS,
    )
    skill_prompt_chars = _get_int(
        yaml_cfg,
        "skill_prompt_chars",
        "AI_AGENT_SKILL_PROMPT_CHARS",
        DEFAULT_SKILL_PROMPT_CHARS,
    )
    command_timeout = _get_int(
        yaml_cfg,
        "command_timeout",
        "AI_AGENT_COMMAND_TIMEOUT",
        DEFAULT_COMMAND_TIMEOUT,
    )
    list_output_limit = _get_int(
        yaml_cfg,
        "list_output_limit",
        "AI_AGENT_LIST_OUTPUT_LIMIT",
        DEFAULT_LIST_OUTPUT_LIMIT,
    )
    include_ignored_files = _get_bool(
        yaml_cfg,
        "include_ignored_files",
        "AI_AGENT_INCLUDE_IGNORED_FILES",
        False,
    )

    if project_root is None:
        project_root_str = (
            os.environ.get("AI_AGENT_PROJECT_ROOT")
            or yaml_cfg.get("project_root")
        )
        if project_root_str:
            project_root = Path(project_root_str).expanduser().resolve()
        else:
            project_root = Path.cwd()

    if skills_root is None:
        skills_root_str = (
            os.environ.get("AI_AGENT_SKILLS_ROOT")
            or yaml_cfg.get("skills_root")
        )
        if skills_root_str:
            skills_root = Path(skills_root_str).expanduser().resolve()
        elif DEFAULT_SKILLS_ROOT.exists():
            skills_root = DEFAULT_SKILLS_ROOT
        else:
            skills_root = None

    return Config(
        model_name=model_name,
        max_context_turns=max_turns,
        max_iterations=max_iterations,
        max_auto_skills=max_auto_skills,
        skill_prompt_chars=skill_prompt_chars,
        command_timeout=command_timeout,
        list_output_limit=list_output_limit,
        include_ignored_files=include_ignored_files,
        project_root=project_root,
        skills_root=skills_root,
    )

