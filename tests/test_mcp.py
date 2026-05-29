"""Wrapping logic only — does not spawn a real MCP server.

For an end-to-end smoke test against Hippocamp, run repl.py with
AMBI_USE_HIPPOCAMP=1.
"""

from dataclasses import dataclass

import pytest

from ambi.integrations.hippocamp import hippocamp_kind_for
from ambi.mcp import mcp_tools


@dataclass
class _StubMcpTool:
    name: str
    description: str
    inputSchema: dict


class _StubServer:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return f"called {name} with {arguments}"


def test_mcp_tools_wraps_each_discovered_tool():
    server = _StubServer()
    discovered = [
        _StubMcpTool("recall_memory", "search memory",
                     {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        _StubMcpTool("update_memory", "write a fact",
                     {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}),
    ]
    wrapped = mcp_tools(server, discovered, kind_for=hippocamp_kind_for)
    assert [t.definition.name for t in wrapped] == ["recall_memory", "update_memory"]
    assert wrapped[0].kind == "read"
    assert wrapped[1].kind == "write"


async def test_wrapped_handler_calls_through_to_server():
    server = _StubServer()
    discovered = [
        _StubMcpTool("recall_memory", "search",
                     {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    ]
    wrapped = mcp_tools(server, discovered)
    result = await wrapped[0].handler({"q": "preferences"})
    assert "recall_memory" in result
    assert server.calls == [("recall_memory", {"q": "preferences"})]


async def test_handler_uses_original_name_after_prefix():
    """name_prefix changes the local name but the remote call uses the original."""
    server = _StubServer()
    discovered = [
        _StubMcpTool("recall_memory", "x",
                     {"type": "object", "properties": {}, "required": []}),
    ]
    wrapped = mcp_tools(server, discovered, name_prefix="hippo_")
    assert wrapped[0].definition.name == "hippo_recall_memory"
    await wrapped[0].handler({})
    # The server was called with the original name, not the prefixed one.
    assert server.calls[0][0] == "recall_memory"


def test_hippocamp_kind_for():
    assert hippocamp_kind_for("recall_memory") == "read"
    assert hippocamp_kind_for("update_memory") == "write"
    assert hippocamp_kind_for("reflect_memory") == "write"
    assert hippocamp_kind_for("anything_else") == "read"


def test_missing_input_schema_falls_back():
    server = _StubServer()
    discovered = [_StubMcpTool("noargs", "x", None)]
    wrapped = mcp_tools(server, discovered)
    schema = wrapped[0].definition.input_schema
    assert schema["type"] == "object"
    assert schema["properties"] == {}
