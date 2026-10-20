"""Persistent MCP client connections for the synchronous agent loop."""

from __future__ import annotations

import asyncio
import atexit
from contextlib import AsyncExitStack
import hashlib
import json
import os
import re
import threading
from typing import Any


def _expand_env(value: Any) -> Any:
    """Expand ${NAME} references in MCP configuration values."""
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda match: os.environ.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _tool_alias(server_name: str, tool_name: str) -> str:
    raw_name = f"mcp__{server_name}__{tool_name}"
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw_name)
    if len(normalized) <= 64:
        return normalized
    suffix = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:53]}_{suffix}"


class MCPClientManager:
    """Keep configured MCP sessions alive on a background asyncio loop."""

    def __init__(self, servers: dict[str, dict[str, Any]] | None = None, timeout: int = 600):
        self.servers = servers or {}
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown: asyncio.Event | None = None
        self._connections: dict[str, Any] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._tool_schemas: list[dict[str, Any]] = []
        self._status: dict[str, dict[str, Any]] = {}
        self._startup_error: str | None = None

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_schemas)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_map)

    def start(self) -> None:
        if not self.servers or self._thread:
            return
        atexit.register(self.close)
        self._thread = threading.Thread(target=self._thread_main, name="mcp-client-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=max(30, 65 * len(self.servers))):
            self._startup_error = "Timed out while connecting to MCP servers."

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self._startup_error = str(exc)
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        for server_name, raw_config in self.servers.items():
            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                client = await asyncio.wait_for(
                    self._connect(server_name, _expand_env(raw_config), stack),
                    timeout=30,
                )
                listed = await asyncio.wait_for(client.list_tools(), timeout=30)
                schemas = []
                server_tools = []
                aliases = set(self._tool_map)
                for tool in listed.tools:
                    alias = _tool_alias(server_name, tool.name)
                    if alias in aliases:
                        raise ValueError(f"MCP tool name collision: {alias}")
                    aliases.add(alias)
                    input_schema = (
                        getattr(tool, "input_schema", None)
                        or getattr(tool, "inputSchema", None)
                        or {"type": "object", "properties": {}}
                    )
                    schemas.append({
                        "type": "function",
                        "function": {
                            "name": alias,
                            "description": tool.description or f"MCP tool {tool.name} from server {server_name}.",
                            "parameters": input_schema,
                        },
                    })
                    server_tools.append((alias, tool.name))
                for alias, remote_name in server_tools:
                    self._tool_map[alias] = (server_name, remote_name)
                self._connections[server_name] = client
                self._stacks[server_name] = stack
                self._tool_schemas.extend(schemas)
                self._status[server_name] = {"connected": True, "tools": len(schemas), "error": None}
            except Exception as exc:
                await stack.aclose()
                self._status[server_name] = {"connected": False, "tools": 0, "error": str(exc)}
        self._ready.set()
        await self._shutdown.wait()
        for stack in reversed(list(self._stacks.values())):
            await stack.aclose()

    async def _connect(self, server_name: str, config: dict[str, Any], stack: AsyncExitStack):
        from mcp import ClientSession, StdioServerParameters

        url = config.get("url")
        command = config.get("command")
        if bool(url) == bool(command):
            raise ValueError("configure exactly one of 'url' or 'command'")

        if url:
            headers = config.get("headers") or {}
            transport = str(config.get("transport") or "streamable_http").lower()
            if transport in ("sse", "sse_http"):
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(
                    sse_client(url, headers=headers, sse_read_timeout=float(self.timeout))
                )
            else:
                from mcp.client.streamable_http import streamable_http_client
                from mcp.shared._httpx_utils import create_mcp_http_client, httpx2

                http_client = await stack.enter_async_context(
                    create_mcp_http_client(
                        headers=headers,
                        timeout=httpx2.Timeout(30, read=float(self.timeout)),
                    )
                )

                read, write = await stack.enter_async_context(
                    streamable_http_client(
                        url,
                        http_client=http_client,
                    )
                )
        else:
            from mcp.client.stdio import stdio_client

            extra_env = config.get("env") or {}
            params = StdioServerParameters(
                command=command,
                args=config.get("args") or [],
                env={**os.environ, **extra_env} if extra_env else None,
                cwd=config.get("cwd"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))

        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=30)
        return session

    def status(self) -> dict[str, dict[str, Any]]:
        if self._startup_error:
            return {"manager": {"connected": False, "tools": 0, "error": self._startup_error}}
        if not self.servers:
            return {}
        return {name: dict(value) for name, value in self._status.items()}

    def has_tool(self, name: str) -> bool:
        return name in self._tool_map

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tool_map:
            return f"Unknown MCP tool: {name}"
        if not self._loop or not self._thread or not self._thread.is_alive():
            return "MCP client is not running. Check /mcp status."
        server_name, remote_name = self._tool_map[name]
        client = self._connections.get(server_name)
        if client is None:
            return f"MCP server {server_name} is disconnected: {self._status.get(server_name, {}).get('error', 'unknown error')}"
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(remote_name, arguments), self._loop
        )
        try:
            return self._format_result(future.result(timeout=self.timeout))
        except Exception as exc:
            future.cancel()
            return f"MCP call failed ({server_name}.{remote_name}): {exc}"

    @staticmethod
    def _format_result(result: Any) -> str:
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            elif getattr(block, "data", None):
                parts.append(f"[{getattr(block, 'type', 'binary')} content returned]")
        structured = getattr(result, "structured_content", None)
        if structured is not None and not parts:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
        if getattr(result, "is_error", False):
            parts.insert(0, "MCP tool reported an error:")
        return "\n".join(parts) if parts else "MCP tool completed with no text content."

    def close(self) -> None:
        if not self._thread or not self._thread.is_alive() or not self._loop or not self._shutdown:
            return
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=10)
