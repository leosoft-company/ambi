"""MCP integration — wrap MCP servers as ambi Tools.

McpServer holds a long-lived stdio MCP connection (use it as an async
context manager so the subprocess shuts down cleanly). `mcp_tools` walks
the server's `list_tools` response and produces ambi Tool instances ready
to register with a ToolRegistry.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Callable, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tool import Tool, ToolKind
from .types import ToolDef


KindFor = Callable[[str], ToolKind]


class McpServer:
    """Async context manager around a stdio MCP server connection."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        errlog: TextIO | str | Path | None = None,
    ):
        self.params = StdioServerParameters(
            command=command,
            args=list(args or []),
            env=env,
        )
        self.errlog = errlog
        self._stack: AsyncExitStack | None = None
        self._owned_errlog: TextIO | None = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "McpServer":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        errlog = self._resolve_errlog()
        read, write = await self._stack.enter_async_context(
            stdio_client(self.params, errlog=errlog)
        )
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()
        return self

    def _resolve_errlog(self) -> TextIO:
        target = self.errlog
        if target is None:
            return sys.stderr
        if isinstance(target, (str, Path)):
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._owned_errlog = path.open("a", encoding="utf-8", buffering=1)
            return self._owned_errlog
        return target

    async def __aexit__(self, *exc_info) -> None:
        try:
            if self._stack is not None:
                await self._stack.__aexit__(*exc_info)
        finally:
            if self._owned_errlog is not None:
                self._owned_errlog.close()
                self._owned_errlog = None
            self._stack = None
            self.session = None

    async def list_tools(self):
        if self.session is None:
            raise RuntimeError("McpServer not entered; use 'async with'")
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        if self.session is None:
            raise RuntimeError("McpServer not entered; use 'async with'")
        result = await self.session.call_tool(name, arguments)
        text = _flatten_content(result.content)
        if getattr(result, "isError", False):
            raise RuntimeError(text or f"MCP tool '{name}' returned isError")
        return text


def _flatten_content(blocks) -> str:
    parts: list[str] = []
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


def mcp_tools(
    server: McpServer,
    discovered,
    kind_for: KindFor | None = None,
    name_prefix: str = "",
) -> list[Tool]:
    """Wrap MCP-discovered tools as ambi Tool instances.

    `discovered` is the list returned by `await server.list_tools()`.
    `kind_for(name)` decides the ToolKind; defaults to 'read' for all.
    `name_prefix` is prepended to the tool name (handy when wiring multiple
    MCP servers that may share names).
    """
    out: list[Tool] = []
    for mt in discovered:
        local_name = f"{name_prefix}{mt.name}"
        kind = kind_for(mt.name) if kind_for else "read"
        # Bind name into the closure so each handler invokes its own tool.
        out.append(_wrap(server, mt, local_name, kind))
    return out


def _wrap(server: McpServer, mt, local_name: str, kind: ToolKind) -> Tool:
    remote_name = mt.name

    async def handler(args: dict) -> str:
        return await server.call_tool(remote_name, args)

    schema = mt.inputSchema or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    return Tool(
        definition=ToolDef(
            name=local_name,
            description=mt.description or f"MCP tool: {remote_name}",
            input_schema=schema,
        ),
        handler=handler,
        kind=kind,
    )
