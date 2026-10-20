from __future__ import annotations

"""
Local web UI for the agent.

Run it with:
    .venv/Scripts/python.exe webui.py [--project-path PATH] [--port 8765]

Then open http://127.0.0.1:8765 in a browser. Everything stays on your
machine: Starlette + uvicorn serve the page, the agent itself is the same
Agent class the CLI uses.
"""

import argparse
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from agent import Agent
from config import load_config
from diffutil import compute_diff
from hooks import HookManager, load_settings
from mcp_client import MCPClientManager
from media import analyze_media, encode_image_base64, get_image_info
from memory import AgentMemory
from models import ModelClient
from sessions import SessionStore
from skills import SkillRegistry
from tools import ProjectTools


WEB_DIR = Path(__file__).parent / "web"
PROJECT_INSTRUCTION_FILES = ["CLAUDE.md", "AGENTS.md", ".agentrules", "AGENT.md"]
UPLOAD_MAX_CHARS = 100_000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
MEDIA_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mkv", ".mov", ".webm"}

# Tool names whose result should carry a structured diff.
MUTATING_WITH_DIFF = {"write_file", "apply_patch", "delete_code", "delete_file", "add_file"}
# ChangeTracker.record() action names, matched by tool name.
ACTIONS_BY_TOOL = {
    "write_file": {"create", "write"},
    "apply_patch": {"patch"},
    "delete_code": {"delete_code"},
    "delete_file": {"delete_file"},
    "add_file": {"create"},
}


class _Stopped(Exception):
    """Raised inside a streaming callback to unwind agent.ask() on /api/stop."""


def _is_tool_protocol_message(content: str) -> bool:
    """
    True for the synthetic messages Agent.ask() injects when the model has no
    native tool calling: the raw {"tool": ...} request it writes, and the
    "Tool result for ..." follow-up. Both are already shown as tool cards via
    the live event stream, so they are noise in a replayed history view.
    """
    stripped = content.strip()
    if stripped.startswith("Tool result for "):
        return True
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and ("tool" in payload or "final" in payload)
    return False


def _load_project_instructions(project_root: Path) -> str:
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


def build_agent(project_path: Optional[Path]) -> Agent:
    cfg = load_config(project_root=project_path.resolve() if project_path else None)
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
    return Agent(
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


class AgentServer:
    """
    Owns the single live Agent instance, the session store, and serializes
    requests to the agent (one turn at a time, like the CLI's REPL).
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.lock = threading.Lock()
        self.sessions = SessionStore(agent.tools.project_root / ".agent_sessions")
        self.uploads_dir = agent.tools.project_root / ".uploads"
        self.session_id = uuid.uuid4().hex[:12]
        self.title = "New chat"
        self.stop_event: Optional[threading.Event] = None

    # Sessions ---------------------------------------------------------
    def autosave(self) -> None:
        has_content = any(
            m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip()
            for m in self.agent.history
        )
        if not has_content:
            return
        if self.title == "New chat":
            first = next(
                (
                    m["content"] for m in self.agent.history
                    if m.get("role") == "user" and isinstance(m.get("content"), str)
                    and not _is_tool_protocol_message(m["content"])
                ),
                "",
            )
            first = first.strip()
            self.title = (first[:60] + "…") if len(first) > 60 else (first or "New chat")
        self.sessions.save(
            self.session_id,
            self.agent.history,
            self.agent.mode,
            self.agent.tasks,
            title=self.title,
            plan_mode=self.agent.plan_mode,
            require_approval=self.agent.require_approval,
            model_name=self.agent.model.model_name,
        )

    def new_session(self) -> None:
        self.autosave()
        self.agent.clear_history()
        self.agent.clear_tasks()
        self.session_id = uuid.uuid4().hex[:12]
        self.title = "New chat"

    def open_session(self, session_id: str) -> None:
        self.autosave()
        data = self.sessions.load(session_id)
        self.agent.history = data.get("history") or self.agent.history[:1]
        self.agent.tasks = data.get("tasks") or []
        if data.get("mode"):
            self.agent.mode = data["mode"]
        self.agent.plan_mode = bool(data.get("plan_mode", False))
        self.agent.require_approval = bool(data.get("require_approval", False))
        stored_model = str(data.get("model_name") or "").strip()
        if stored_model:
            self.agent.model.model_name = stored_model
        self.session_id = str(data.get("id", session_id))
        self.title = str(data.get("title") or "New chat")

    # Uploads ------------------------------------------------------------
    def ingest_upload(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            try:
                b64, mime = encode_image_base64(str(path))
                desc = self.agent.model.describe_image(
                    image_base64=b64, prompt="Describe this image in detail.", mime_type=mime
                )
                meta = get_image_info(str(path))
                return f"[image] {meta}\n\nAI description:\n{desc}"
            except Exception as exc:  # noqa: BLE001
                return f"[image] {path.name} (vision analysis failed: {exc})"
        if ext in MEDIA_EXTS:
            try:
                return f"[media]\n{analyze_media(str(path))}"
            except Exception as exc:  # noqa: BLE001
                return f"[media] {path.name} (analysis failed: {exc})"
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

    # Diffs ----------------------------------------------------------------
    def _diff_for(self, name: str, args: dict) -> Optional[dict]:
        if name not in MUTATING_WITH_DIFF:
            return None
        path_arg = args.get("path") or ""
        wanted_actions = ACTIONS_BY_TOOL.get(name, set())
        entry = None
        for e in reversed(self.agent.tools.changes.entries):
            if e.action in wanted_actions and (e.path == path_arg or path_arg in e.path):
                entry = e
                break
        if entry is None:
            return None
        if name == "apply_patch":
            diff = compute_diff(
                pre_text=entry.pre_text,
                post_text=entry.post_text,
                old_snippet=entry.old_snippet,
                new_snippet=entry.new_snippet,
                old_start_line=entry.old_start_line,
            )
        elif name == "delete_code":
            diff = compute_diff(
                pre_text=entry.pre_text,
                post_text=entry.post_text,
                old_snippet=entry.old_snippet,
                new_snippet="",
                old_start_line=entry.old_start_line,
            )
        else:
            diff = compute_diff(pre_text=entry.pre_text, post_text=entry.post_text)
        return {"path": entry.path, "action": entry.action, **diff.to_dict()}

    # Turn execution -------------------------------------------------------
    def run(self, message: str, events: "queue.Queue[dict]", stop_event: threading.Event) -> None:
        def check_stop() -> None:
            if stop_event.is_set():
                raise _Stopped()

        def on_token(token: str) -> None:
            check_stop()
            events.put({"type": "token", "text": token})

        def on_tool_call(name: str, args: dict, result: str) -> None:
            events.put({
                "type": "tool",
                "name": name,
                "args": args,
                "result": result[:4000],
                "diff": self._diff_for(name, args),
            })
            check_stop()

        def on_approval(name: str, detail: str) -> bool:
            # The web UI has no synchronous prompt yet; approve-gated tools
            # are reported but not blocked. Use /approve off (the default).
            return True

        try:
            reply = self.agent.ask(
                message,
                on_tool_call=on_tool_call,
                on_approval=on_approval,
                on_token=on_token,
            )
            events.put({"type": "done", "reply": reply})
            self.autosave()
        except _Stopped:
            events.put({"type": "stopped"})
            self.autosave()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the browser
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put({"type": "__end__"})


def create_app(server: AgentServer) -> Starlette:
    async def index(request: Request) -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    def _history_payload() -> dict:
        agent = server.agent
        messages = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in agent.history
            if m.get("role") in {"user", "assistant"}
            and isinstance(m.get("content"), str)
            and not _is_tool_protocol_message(m.get("content"))
        ]
        return {
            "messages": messages,
            "tasks": agent.tasks,
            "mode": agent.mode,
            "model": agent.model.model_name,
            "routing_mode": agent.model.routing_mode,
            "fast_model": agent.model.fast_model,
            "reasoning_model": agent.model.reasoning_model,
            "last_routed_model": agent.model.last_routed_model,
            "project_root": str(agent.tools.project_root),
            "session_id": server.session_id,
            "title": server.title,
        }

    async def api_history(request: Request) -> JSONResponse:
        return JSONResponse(_history_payload())

    async def api_send(request: Request) -> EventSourceResponse:
        body = await request.json()
        message = (body.get("message") or "").strip()

        async def event_stream():
            if not message:
                yield {"event": "error", "data": json.dumps({"message": "Empty message."})}
                return
            if not server.lock.acquire(blocking=False):
                yield {"event": "error", "data": json.dumps({"message": "Agent is busy with another request."})}
                return
            events: "queue.Queue[dict]" = queue.Queue()
            stop_event = threading.Event()
            server.stop_event = stop_event
            worker = threading.Thread(target=server.run, args=(message, events, stop_event), daemon=True)
            worker.start()
            try:
                while True:
                    item = await anyio.to_thread.run_sync(events.get)
                    if item["type"] == "__end__":
                        break
                    yield {"event": item["type"], "data": json.dumps(item)}
            finally:
                server.stop_event = None
                server.lock.release()

        return EventSourceResponse(event_stream())

    async def api_chat(request: Request) -> JSONResponse:
        """
        Compatibility route for web/index.js, which only ever calls
        /api/chat and /api/chat/stream (the web/server.py Flask contract).
        Without these, a POST here falls through to the static file Mount
        below and Starlette answers with 405 Method Not Allowed.
        """
        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "No message provided"}, status_code=400)
        if not server.lock.acquire(blocking=False):
            return JSONResponse({"error": "Agent is busy with another request."}, status_code=409)
        try:
            reply = await anyio.to_thread.run_sync(server.agent.ask, message)
            server.autosave()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            server.lock.release()
        return JSONResponse({
            "response": reply,
            "mode": server.agent.mode,
            "timestamp": time.time(),
        })

    async def api_chat_stream(request: Request) -> EventSourceResponse:
        """Same compatibility shim as api_chat, in the SSE shape index.js expects."""
        body = await request.json()
        message = (body.get("message") or "").strip()

        async def event_stream():
            if not message:
                yield {"data": json.dumps({"type": "error", "message": "No message provided"})}
                return
            if not server.lock.acquire(blocking=False):
                yield {"data": json.dumps({"type": "error", "message": "Agent is busy with another request."})}
                return
            events: "queue.Queue[dict]" = queue.Queue()
            stop_event = threading.Event()
            server.stop_event = stop_event
            worker = threading.Thread(target=server.run, args=(message, events, stop_event), daemon=True)
            worker.start()
            yield {"data": json.dumps({"type": "start"})}
            try:
                while True:
                    item = await anyio.to_thread.run_sync(events.get)
                    kind = item["type"]
                    if kind == "__end__":
                        break
                    if kind == "token":
                        yield {"data": json.dumps({"type": "chunk", "content": item["text"]})}
                    elif kind == "error":
                        yield {"data": json.dumps({"type": "error", "message": item.get("message", "")})}
                yield {"data": json.dumps({"type": "end"})}
            finally:
                server.stop_event = None
                server.lock.release()

        return EventSourceResponse(event_stream())

    async def api_stop(request: Request) -> JSONResponse:
        ev = server.stop_event
        if ev is None:
            return JSONResponse({"stopped": False, "message": "Nothing is running."})
        ev.set()
        return JSONResponse({"stopped": True})

    async def api_sessions(request: Request) -> JSONResponse:
        rows = server.sessions.list_sessions()
        for row in rows:
            row["current"] = row.get("id") == server.session_id
        return JSONResponse({"sessions": rows})

    async def api_session_new(request: Request) -> JSONResponse:
        if not server.lock.acquire(blocking=False):
            return JSONResponse({"error": "Agent is busy."}, status_code=409)
        try:
            server.new_session()
        finally:
            server.lock.release()
        return JSONResponse(_history_payload())

    async def api_session_open(request: Request) -> JSONResponse:
        session_id = request.path_params["session_id"]
        if not server.lock.acquire(blocking=False):
            return JSONResponse({"error": "Agent is busy."}, status_code=409)
        try:
            server.open_session(session_id)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        finally:
            server.lock.release()
        return JSONResponse(_history_payload())

    async def api_session_rename(request: Request) -> JSONResponse:
        session_id = request.path_params["session_id"]
        body = await request.json()
        title = (body.get("title") or "").strip() or "New chat"
        try:
            server.sessions.rename(session_id, title)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        if session_id == server.session_id:
            server.title = title
        return JSONResponse({"ok": True})

    async def api_session_delete(request: Request) -> JSONResponse:
        session_id = request.path_params["session_id"]
        deleted = server.sessions.delete(session_id)
        return JSONResponse({"deleted": deleted})

    async def api_models(request: Request) -> JSONResponse:
        try:
            models = server.agent.model.list_available_models()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"models": [], "error": str(exc)})
        return JSONResponse({"models": models, "current": server.agent.model.model_name})

    async def api_model_set(request: Request) -> JSONResponse:
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "Missing model name."}, status_code=400)
        server.agent.model.model_name = name
        return JSONResponse({"model": name})

    async def api_routing_set(request: Request) -> JSONResponse:
        body = await request.json()
        model = server.agent.model
        if "routing_mode" in body:
            mode = (body.get("routing_mode") or "single").strip().lower()
            if mode not in {"single", "hybrid"}:
                return JSONResponse({"error": "routing_mode must be 'single' or 'hybrid'."}, status_code=400)
            model.routing_mode = mode
        if "fast_model" in body:
            model.fast_model = (body.get("fast_model") or "").strip()
        if "reasoning_model" in body:
            model.reasoning_model = (body.get("reasoning_model") or "").strip()
        return JSONResponse({
            "routing_mode": model.routing_mode,
            "fast_model": model.fast_model,
            "reasoning_model": model.reasoning_model,
        })

    async def api_upload(request: Request) -> JSONResponse:
        if not server.lock.acquire(blocking=False):
            return JSONResponse({"error": "Agent is busy."}, status_code=409)
        try:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "No file provided."}, status_code=400)
            raw = await upload.read()
            safe_name = re.sub(r"[^\w.\-]+", "_", upload.filename or "upload")
            server.uploads_dir.mkdir(parents=True, exist_ok=True)
            dest = server.uploads_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
            dest.write_bytes(raw)
            block = server.ingest_upload(dest)
            server.agent.history.append({
                "role": "user",
                "content": f"[Uploaded file: {safe_name}]\n{block}",
            })
            return JSONResponse({"filename": safe_name, "chars": len(block)})
        finally:
            server.lock.release()

    routes = [
        Route("/", index),
        Route("/api/history", api_history),
        Route("/api/send", api_send, methods=["POST"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Route("/api/chat/stream", api_chat_stream, methods=["POST"]),
        Route("/api/stop", api_stop, methods=["POST"]),
        Route("/api/sessions", api_sessions),
        Route("/api/sessions/new", api_session_new, methods=["POST"]),
        Route("/api/sessions/{session_id}/open", api_session_open, methods=["POST"]),
        Route("/api/sessions/{session_id}/rename", api_session_rename, methods=["POST"]),
        Route("/api/sessions/{session_id}", api_session_delete, methods=["DELETE"]),
        Route("/api/models", api_models),
        Route("/api/model", api_model_set, methods=["POST"]),
        Route("/api/routing", api_routing_set, methods=["POST"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        # CSS/JS are referenced as /styles.css and /index.js in index.html.
        # Mount last so /api/* and / still match the routes above.
        Mount("/", StaticFiles(directory=WEB_DIR), name="static"),
    ]
    return Starlette(routes=routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for the AI agent.")
    parser.add_argument(
        "--project-path",
        type=Path,
        default=None,
        help="Project directory for the agent. Defaults to this script's own "
        "directory, NOT the current working directory — webui.py is often "
        "launched from a shortcut or batch file whose cwd can be anywhere "
        "(even a protected folder like C:\\Windows\\System32), which would "
        "otherwise make the agent try to write its session/memory files there.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    project_path = args.project_path or Path(__file__).resolve().parent
    agent = build_agent(project_path)
    server = AgentServer(agent)
    app = create_app(server)

    print(f"Agent web UI: http://{args.host}:{args.port}  (project: {agent.tools.project_root})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
