from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from agent import Agent
from config import load_config
from models import ModelClient
from skills import SkillRegistry
from tools import ProjectTools, ChangeTracker, fetch_url, search_web


app = typer.Typer(help="Local AI code helper agent")
console = Console()


def _print_help() -> None:
    console.print(
        "[bold]Commands[/bold]\n"
        "  /exit                 Exit the agent\n"
        "  /help                 Show this help\n"
        "  /http <url>           Fetch a URL over HTTP/HTTPS and show the text response\n"
        "  /search <query>       Search the web using DuckDuckGo\n"
        "  /delete <path>        Delete a file from the project\n"
        "  /mkdir <path>         Create a directory in the project\n"
        "  /changes              Show all file changes made this session\n"
        "  /changes clear        Clear the changes log\n"
        "  /skills \\[query]       List skills, or search for matching skills\n"
        "  /use <skill>          Keep a skill loaded for future requests\n"
        "  /clear-skills         Clear manually loaded skills\n"
        "  /auto-skills on|off   Enable or disable automatic skill matching\n"
        "  /modes                List available agent modes\n"
        "  /mode <name>          Switch mode: coder, reviewer, debugger, architect...\n"
        "  /limits               Show current capacity settings\n"
        "\n"
        "Anything else is sent to the AI assistant, which can automatically\n"
        "load relevant skills, read/write/create/delete files, run commands,\n"
        "and debug your code. Use /changes to review what the AI modified.\n"
    )


def _on_tool_call(name: str, args: dict, result: str) -> None:
    """Callback to display tool usage in the terminal."""
    if name == "read_file":
        console.print(f"  [dim]📖 Reading:[/dim] {args.get('path', '?')}")
    elif name == "list_files":
        subdir = args.get("subdir", ".")
        console.print(f"  [dim]📁 Listing:[/dim] {subdir or '.'}")
    elif name == "list_skill_files":
        console.print(f"  [dim]Skill files:[/dim] {args.get('skill_name', '?')}")
    elif name == "read_skill_file":
        console.print(
            f"  [dim]Skill read:[/dim] "
            f"{args.get('skill_name', '?')}/{args.get('path', '?')}"
        )
    elif name == "write_file":
        console.print(f"  [dim]✏️  Writing:[/dim] {args.get('path', '?')}")
    elif name == "apply_patch":
        console.print(f"  [dim]🩹 Patching:[/dim] {args.get('path', '?')}")
    elif name == "delete_file":
        console.print(f"  [dim]🗑️  Deleting:[/dim] {args.get('path', '?')}")
    elif name == "delete_code":
        console.print(f"  [dim]✂️  Removing code from:[/dim] {args.get('path', '?')}")
    elif name == "create_directory":
        console.print(f"  [dim]📂 Creating dir:[/dim] {args.get('path', '?')}")
    elif name == "run_command":
        cmd = args.get("command", "?")
        console.print(f"  [dim]▶️  Running:[/dim] {cmd}")
        # Show command output (truncated for display)
        output_lines = result.strip().split("\n")
        for line in output_lines[:15]:
            console.print(f"  [dim]   {line}[/dim]")
        if len(output_lines) > 15:
            console.print(f"  [dim]   ... ({len(output_lines) - 15} more lines)[/dim]")
    elif name == "fetch_url":
        console.print(f"  [dim]Internet:[/dim] {args.get('url', '?')}")
        output_lines = result.strip().split("\n")
        for line in output_lines[:8]:
            console.print(f"  [dim]   {line[:160]}[/dim]")
        if len(output_lines) > 8:
            console.print(f"  [dim]   ... ({len(output_lines) - 8} more lines)[/dim]")
    elif name == "search_web":
        console.print(f"  [dim]🔍 Searching web:[/dim] {args.get('query', '?')}")
        output_lines = result.strip().split("\n")
        for line in output_lines[:8]:
            console.print(f"  [dim]   {line[:160]}[/dim]")
        if len(output_lines) > 8:
            console.print(f"  [dim]   ... ({len(output_lines) - 8} more lines)[/dim]")
    else:
        console.print(f"  [dim]🔧 {name}[/dim]")


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
        help="Name of the local model to use (e.g. llama3).",
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
    skills_path: Optional[Path] = typer.Option(
        None,
        "--skills-path",
        help="Path to a directory containing Claude-style skills.",
    ),
) -> None:
    """
    Start an interactive REPL with the local AI agent.
    """
    cfg_project_root = project_path.resolve() if project_path else None
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
    cfg.include_ignored_files = include_ignored_files

    tools = ProjectTools(
        project_root=cfg.project_root,
        command_timeout=cfg.command_timeout,
        include_ignored_files=cfg.include_ignored_files,
    )
    model_client = ModelClient(model_name=cfg.model_name)
    skill_registry = SkillRegistry(cfg.skills_root)
    agent = Agent(
        model=model_client,
        tools=tools,
        skills=skill_registry,
        max_context_turns=cfg.max_context_turns,
        max_iterations=cfg.max_iterations,
        max_auto_skills=cfg.max_auto_skills,
        skill_prompt_chars=cfg.skill_prompt_chars,
    )

    if not cfg.project_root.exists():
        console.print(
            f"[red]Project path does not exist:[/red] {cfg.project_root}\n"
            "Start again with --project-path C:/path/to/your/project or update config.yaml."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold green]AI agent started[/bold green] "
        f"(model: [bold]{cfg.model_name}[/bold], "
        f"project: [bold]{cfg.project_root}[/bold])."
    )
    if skill_registry:
        console.print(
            f"[green]Loaded {len(skill_registry)} skills[/green] "
            f"from [bold]{skill_registry.skills_root}[/bold]."
        )
    else:
        console.print("[yellow]No skills loaded. Use --skills-path to add them.[/yellow]")
    console.print(
        "[dim]Power mode: "
        f"{cfg.max_iterations} iterations, "
        f"{cfg.max_context_turns} context turns, "
        f"{cfg.command_timeout}s commands.[/dim]"
    )
    _print_help()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting.")
            break

        stripped = user_input.strip().lstrip("\ufeff")
        if not stripped:
            continue

        if stripped in {"/exit", ":q"}:
            console.print("Goodbye.")
            break

        if stripped in {"/help", ":h"}:
            _print_help()
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

        if stripped == "/limits":
            _print_limits(cfg)
            continue

        if stripped == "/mode" or stripped.startswith("/mode "):
            if stripped == "/mode":
                console.print(f"Current mode: [bold]{agent.mode}[/bold]")
                continue
            mode_name = stripped.split(maxsplit=1)[1]
            try:
                agent.set_mode(mode_name)
                console.print(f"[green]Mode switched to:[/green] {agent.mode}")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        # Normal conversation — agent uses tools automatically
        skill_names = agent.preview_skill_names(stripped)
        if skill_names:
            console.print(f"[dim]Using skills: {', '.join(skill_names)}[/dim]")
        console.print(f"[dim]Thinking in {agent.mode} mode...[/dim]")
        reply = agent.ask(user_input, on_tool_call=_on_tool_call)
        console.print(f"\n[bold magenta]Agent[/bold magenta]: {reply}")

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


if __name__ == "__main__":
    app()
