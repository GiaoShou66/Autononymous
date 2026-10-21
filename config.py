from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_MODEL_NAME = "cryptidbleh/gemma4-claude-opus-4.6:latest"
DEFAULT_MODE = "unrestricted"
DEFAULT_MAX_CONTEXT_TURNS = 1000
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_MAX_AUTO_SKILLS = 5
DEFAULT_SKILL_PROMPT_CHARS = 25000
DEFAULT_COMMAND_TIMEOUT = 3600
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_LIST_OUTPUT_LIMIT = 0
DEFAULT_RESTRICT_TO_PROJECT = False
DEFAULT_MEMORY_MAX_LESSONS = 200
DEFAULT_COMPACT_THRESHOLD = 40
DEFAULT_LOOP_MAX_CYCLES = 10
DEFAULT_LOOP_INTERVAL_SECONDS = 0.0
DEFAULT_REASONING_ENABLED = True
DEFAULT_ROUTING_MODE = "single"
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_SKILLS_ROOT = Path(__file__).parent.resolve() / "skills"


@dataclass
class Config:
    model_name: str = DEFAULT_MODEL_NAME
    mode: str = DEFAULT_MODE
    max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_auto_skills: int = DEFAULT_MAX_AUTO_SKILLS
    skill_prompt_chars: int = DEFAULT_SKILL_PROMPT_CHARS
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    list_output_limit: int = DEFAULT_LIST_OUTPUT_LIMIT
    include_ignored_files: bool = True
    restrict_to_project: bool = DEFAULT_RESTRICT_TO_PROJECT
    project_root: Path = Path.cwd()
    skills_root: Optional[Path] = DEFAULT_SKILLS_ROOT
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    memory_path: Optional[Path] = None
    memory_max_lessons: int = DEFAULT_MEMORY_MAX_LESSONS
    compact_threshold: int = DEFAULT_COMPACT_THRESHOLD
    loop_max_cycles: int = DEFAULT_LOOP_MAX_CYCLES
    loop_interval_seconds: float = DEFAULT_LOOP_INTERVAL_SECONDS
    reasoning_enabled: bool = DEFAULT_REASONING_ENABLED
    routing_mode: str = DEFAULT_ROUTING_MODE
    fast_model: str = ""
    reasoning_model: str = ""
    commandcode_api_key: str = ""
    deepseek_api_key: str = ""
    custom_model_url: str = ""
    custom_model_key: str = ""


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


def _load_dotenv_local(project_dir: Path) -> None:
    """
    Fill os.environ from a .env.local file (KEY=VALUE per line, # comments,
    blank lines ignored) sitting next to config.yaml. Never overrides a
    variable that's already set — a real environment variable (e.g. one set
    with setx) still wins. .env.local is git-ignored; this exists because
    Windows env vars set with setx only reach processes started fresh AFTER
    the call, in a terminal also opened after it — an already-running shell
    (or one spawned from an IDE that predates the setx call) never sees it,
    which is a common, confusing failure mode. A file read at process start
    has no such inheritance gotcha.
    """
    path = project_dir / ".env.local"
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def load_config(
    config_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> Config:
    """
    Load configuration from (in order of precedence):
    - Explicit function arguments
    - Environment variables (including ones filled from .env.local)
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

    _load_dotenv_local(config_path.parent if config_path.is_file() else Path(__file__).parent.resolve())

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
    request_timeout = _get_int(
        yaml_cfg,
        "request_timeout",
        "AI_AGENT_REQUEST_TIMEOUT",
        DEFAULT_REQUEST_TIMEOUT,
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
        True,
    )
    restrict_to_project = _get_bool(
        yaml_cfg,
        "restrict_to_project",
        "AI_AGENT_RESTRICT_TO_PROJECT",
        DEFAULT_RESTRICT_TO_PROJECT,
    )
    reasoning_enabled = _get_bool(
        yaml_cfg,
        "reasoning_enabled",
        "AI_AGENT_REASONING",
        DEFAULT_REASONING_ENABLED,
    )
    routing_mode = (
        os.environ.get("AI_AGENT_ROUTING_MODE", "").strip().lower()
        or str(yaml_cfg.get("routing_mode") or "").strip().lower()
        or DEFAULT_ROUTING_MODE
    )
    if routing_mode not in {"single", "hybrid"}:
        routing_mode = DEFAULT_ROUTING_MODE
    fast_model = (
        os.environ.get("AI_AGENT_FAST_MODEL", "").strip()
        or str(yaml_cfg.get("fast_model") or "").strip()
    )
    reasoning_model = (
        os.environ.get("AI_AGENT_REASONING_MODEL", "").strip()
        or str(yaml_cfg.get("reasoning_model") or "").strip()
    )
    mode = (
        os.environ.get("AI_AGENT_MODE", "").strip()
        or (yaml_cfg.get("mode") or "").strip()
        or DEFAULT_MODE
    )
    memory_max_lessons = _get_int(
        yaml_cfg,
        "memory_max_lessons",
        "AI_AGENT_MEMORY_MAX_LESSONS",
        DEFAULT_MEMORY_MAX_LESSONS,
    )
    compact_threshold = _get_int(
        yaml_cfg,
        "compact_threshold",
        "AI_AGENT_COMPACT_THRESHOLD",
        DEFAULT_COMPACT_THRESHOLD,
    )
    loop_max_cycles = _get_int(
        yaml_cfg,
        "loop_max_cycles",
        "AI_AGENT_LOOP_MAX_CYCLES",
        DEFAULT_LOOP_MAX_CYCLES,
    )
    _loop_interval_raw = (
        os.environ.get("AI_AGENT_LOOP_INTERVAL")
        or yaml_cfg.get("loop_interval_seconds")
    )
    try:
        loop_interval_seconds = (
            float(_loop_interval_raw)
            if _loop_interval_raw is not None
            else DEFAULT_LOOP_INTERVAL_SECONDS
        )
    except (TypeError, ValueError):
        loop_interval_seconds = DEFAULT_LOOP_INTERVAL_SECONDS
    memory_path_str = (
        os.environ.get("AI_AGENT_MEMORY_PATH")
        or yaml_cfg.get("memory_path")
    )
    memory_path = (
        Path(memory_path_str).expanduser().resolve() if memory_path_str else None
    )
    raw_mcp_servers = yaml_cfg.get("mcp_servers") or {}
    mcp_servers = (
        {name: server for name, server in raw_mcp_servers.items() if isinstance(server, dict)}
        if isinstance(raw_mcp_servers, dict)
        else {}
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
            skills_root = Path(skills_root_str).expanduser()
            if not skills_root.is_absolute():
                skills_root = project_root / skills_root
            skills_root = skills_root.resolve()
        elif DEFAULT_SKILLS_ROOT.exists():
            skills_root = DEFAULT_SKILLS_ROOT
        else:
            skills_root = None

    commandcode_api_key = (
        os.environ.get("COMMANDCODE_API_KEY", "").strip()
        or str(yaml_cfg.get("commandcode_api_key") or "").strip()
    )
    if commandcode_api_key:
        os.environ["COMMANDCODE_API_KEY"] = commandcode_api_key

    deepseek_api_key = (
        os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or str(yaml_cfg.get("deepseek_api_key") or "").strip()
    )
    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

    custom_model_url = (
        os.environ.get("CUSTOM_MODEL_URL", "").strip()
        or str(yaml_cfg.get("custom_model_url") or "").strip()
    )
    if custom_model_url:
        os.environ["CUSTOM_MODEL_URL"] = custom_model_url

    custom_model_key = (
        os.environ.get("CUSTOM_MODEL_KEY", "").strip()
        or str(yaml_cfg.get("custom_model_key") or "").strip()
    )
    if custom_model_key:
        os.environ["CUSTOM_MODEL_KEY"] = custom_model_key

    return Config(
        model_name=model_name,
        mode=mode,
        max_context_turns=max_turns,
        max_iterations=max_iterations,
        max_auto_skills=max_auto_skills,
        skill_prompt_chars=skill_prompt_chars,
        command_timeout=command_timeout,
        request_timeout=request_timeout,
        list_output_limit=list_output_limit,
        include_ignored_files=include_ignored_files,
        restrict_to_project=restrict_to_project,
        project_root=project_root,
        skills_root=skills_root,
        mcp_servers=mcp_servers,
        memory_path=memory_path,
        memory_max_lessons=memory_max_lessons,
        compact_threshold=compact_threshold,
        loop_max_cycles=loop_max_cycles,
        loop_interval_seconds=loop_interval_seconds,
        reasoning_enabled=reasoning_enabled,
        routing_mode=routing_mode,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
        commandcode_api_key=commandcode_api_key,
        deepseek_api_key=deepseek_api_key,
        custom_model_url=custom_model_url,
        custom_model_key=custom_model_key,
    )

