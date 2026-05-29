"""Hippocamp — portable, persistent memory exposed via MCP.

Hippocamp's tools (as of writing) include:

    recall_memory      — search memory by query                       (read)
    update_memory      — write a fact/preference/observation          (write)
    reflect_memory     — generate higher-level reflection             (write)

The classifier below assumes any tool starting with `update` or `reflect`
is a write; everything else is a read. Override `kind_for` if you want
different semantics.

Typical usage:

    async with hippocamp_server() as hippo:
        tools = await load_hippocamp_tools(hippo)
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)
        agent = Agent(provider=..., tools=registry, ...)
        # ... chat loop runs inside the `async with` so the subprocess lives
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from ..mcp import KindFor, McpServer, mcp_tools
from ..tool import Tool, ToolKind


def hippocamp_kind_for(name: str) -> ToolKind:
    if name.startswith("update") or name.startswith("reflect"):
        return "write"
    return "read"


def hippocamp_server(
    command: str = "hippocamp-mcp",
    args: list[str] | None = None,
    errlog: TextIO | str | Path | None = None,
) -> McpServer:
    """Build (but don't start) an MCP server handle for Hippocamp.

    The default `hippocamp-mcp` is the entry point installed by
    ``pip install hippocamp`` (or `ambi-core[hippocamp]`). Override
    command/args if your install lives elsewhere.

    Pass ``errlog`` (file path or open TextIO) to redirect Hippocamp's
    stderr — useful because its embedding library prints tqdm progress
    bars on every batch.
    """
    return McpServer(command=command, args=args, errlog=errlog)


async def load_hippocamp_tools(
    server: McpServer,
    kind_for: KindFor | None = None,
) -> list[Tool]:
    """Discover Hippocamp's MCP tools and wrap them as ambi Tools."""
    discovered = await server.list_tools()
    return mcp_tools(
        server, discovered, kind_for=kind_for or hippocamp_kind_for,
    )
