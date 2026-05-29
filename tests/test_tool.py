import pytest

from ambi.tool import Tool, ToolRegistry
from ambi.types import ToolDef


def _make_tool(name: str, handler) -> Tool:
    return Tool(
        definition=ToolDef(
            name=name,
            description=f"{name} tool",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        handler=handler,
    )


def test_register_and_defs():
    reg = ToolRegistry()

    async def h(args):
        return "ok"

    reg.register(_make_tool("a", h))
    reg.register(_make_tool("b", h))
    names = [d.name for d in reg.defs()]
    assert set(names) == {"a", "b"}


async def test_invoke_success_sets_tool_name():
    reg = ToolRegistry()

    async def h(args):
        return f"got {args['x']}"

    reg.register(_make_tool("echo", h))
    res = await reg.invoke("echo", {"x": 42})
    assert res.content == "got 42"
    assert res.is_error is False
    assert res._tool_name == "echo"


async def test_invoke_unknown_tool_returns_error_with_available_list():
    reg = ToolRegistry()

    async def h(args):
        return "ok"

    reg.register(_make_tool("known_one", h))
    res = await reg.invoke("ghost", {})
    assert res.is_error is True
    assert "not registered" in res.content
    assert "known_one" in res.content


def test_kind_of_unknown_tool_defaults_to_read():
    reg = ToolRegistry()
    assert reg.kind("anything") == "read"


async def test_invoke_handler_exception_becomes_error_result():
    reg = ToolRegistry()

    async def boom(args):
        raise ValueError("nope")

    reg.register(_make_tool("boom", boom))
    res = await reg.invoke("boom", {})
    assert res.is_error is True
    assert "nope" in res.content
    assert res._tool_name == "boom"
