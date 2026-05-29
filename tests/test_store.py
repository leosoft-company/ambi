import pytest

from ambi.store import SqliteStore
from ambi.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def _msg(role, *blocks):
    return Message(role=role, content=list(blocks))


async def test_empty_load_returns_empty(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    assert await store.load() == []


async def test_append_and_load_roundtrip(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    msgs = [
        _msg("user", TextBlock("hello")),
        _msg("assistant", TextBlock("hi back")),
    ]
    await store.append(msgs)
    loaded = await store.load()
    assert loaded == msgs


async def test_all_block_types_roundtrip(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    msgs = [
        _msg("user", TextBlock("please calculate")),
        _msg(
            "assistant",
            TextBlock("calling tool"),
            ToolUseBlock(id="t1", name="add", input={"a": 1, "b": 2}),
        ),
        _msg(
            "user",
            ToolResultBlock(
                tool_use_id="t1",
                content="3",
                is_error=False,
                _tool_name="add",
            ),
        ),
        _msg("assistant", TextBlock("answer is 3")),
    ]
    await store.append(msgs)
    loaded = await store.load()
    assert loaded == msgs
    # Specific check that the hidden _tool_name survives serialization.
    assert loaded[2].content[0]._tool_name == "add"


async def test_append_is_incremental(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    await store.append([_msg("user", TextBlock("first"))])
    await store.append([_msg("assistant", TextBlock("second"))])
    await store.append([_msg("user", TextBlock("third"))])
    loaded = await store.load()
    texts = [m.content[0].text for m in loaded]
    assert texts == ["first", "second", "third"]


async def test_sessions_isolated(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    await store.append([_msg("user", TextBlock("a"))], session_id="alpha")
    await store.append([_msg("user", TextBlock("b"))], session_id="beta")

    alpha = await store.load("alpha")
    beta = await store.load("beta")
    assert [m.content[0].text for m in alpha] == ["a"]
    assert [m.content[0].text for m in beta] == ["b"]


async def test_clear_removes_session_only(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    await store.append([_msg("user", TextBlock("a"))], session_id="alpha")
    await store.append([_msg("user", TextBlock("b"))], session_id="beta")
    await store.clear("alpha")
    assert await store.load("alpha") == []
    assert len(await store.load("beta")) == 1


async def test_empty_append_is_noop(tmp_path):
    store = SqliteStore(tmp_path / "session.db")
    await store.append([])
    assert await store.load() == []


async def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "ambi.db"
    store = SqliteStore(nested)
    await store.append([_msg("user", TextBlock("hi"))])
    assert nested.exists()
    assert len(await store.load()) == 1


