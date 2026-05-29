import asyncio

import pytest

from ambi.loop import Agent, MaxTurnsExceeded
from ambi.sensegate import SenseGate, Verdict
from ambi.skills import SkillRegistry
from ambi.store import SqliteStore
from ambi.tool import Tool, ToolKind, ToolRegistry
from ambi.types import (
    CompletionResult,
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)

from tests.mock_provider import MockProvider


def _tool(name: str, handler, kind: ToolKind = "read") -> Tool:
    return Tool(
        definition=ToolDef(
            name=name,
            description=f"{name} tool",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        handler=handler,
        kind=kind,
    )


class _ScriptedVerifier:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    async def verify(self, final_text, invocations):
        self.calls.append({"final_text": final_text, "invocations": list(invocations)})
        return self.verdicts.pop(0)


async def test_end_turn_returns_immediately():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn")]
    )
    agent = Agent(provider=provider, tools=ToolRegistry(), system="be nice")
    text = await agent.chat("hello")
    assert text == "hi"
    assert len(agent.messages) == 2
    assert agent.messages[0].role == "user"
    assert agent.messages[1].role == "assistant"
    assert provider.calls[0]["system"] == "be nice"


async def test_chat_accumulates_history_across_calls():
    provider = MockProvider(
        [
            CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("again")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s")
    await agent.chat("first")
    await agent.chat("second")
    # Second call should see history from the first.
    sent_on_second = provider.calls[1]["messages"]
    assert len(sent_on_second) == 3  # user(first), assistant(hi), user(second)
    assert sent_on_second[2].content[0].text == "second"


async def test_tool_call_then_final_response():
    tools = ToolRegistry()

    async def add(args):
        return str(args["a"] + args["b"])

    tools.register(_tool("add", add))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="t1", name="add", input={"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            CompletionResult(
                content=[TextBlock("the answer is 3")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="math agent")
    text = await agent.chat("what's 1+2?")

    assert text == "the answer is 3"
    assert len(agent.messages) == 4
    tool_result_msg = agent.messages[2]
    block = tool_result_msg.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "t1"
    assert block._tool_name == "add"
    assert block.content == "3"
    assert block.is_error is False


async def test_parallel_tool_calls_dispatched_concurrently():
    tools = ToolRegistry()
    started: list[str] = []
    release = asyncio.Event()

    async def slow(args):
        started.append(args["id"])
        await release.wait()
        return f"done {args['id']}"

    tools.register(_tool("slow", slow))

    provider = MockProvider(
        [
            CompletionResult(
                content=[
                    ToolUseBlock(id="a", name="slow", input={"id": "a"}),
                    ToolUseBlock(id="b", name="slow", input={"id": "b"}),
                ],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s")
    task = asyncio.create_task(agent.chat("go"))
    for _ in range(10):
        await asyncio.sleep(0)
        if len(started) >= 2:
            break
    assert started == ["a", "b"]
    release.set()
    await task

    results = agent.messages[2].content
    assert [r.tool_use_id for r in results] == ["a", "b"]
    assert [r.content for r in results] == ["done a", "done b"]


async def test_handler_exception_propagates_as_error_result():
    tools = ToolRegistry()

    async def boom(args):
        raise RuntimeError("kaboom")

    tools.register(_tool("boom", boom))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="x", name="boom", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("noted")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s")
    await agent.chat("trigger error")
    err = agent.messages[2].content[0]
    assert err.is_error is True
    assert "kaboom" in err.content


async def test_hits_max_turns_raises_and_rolls_back():
    tools = ToolRegistry()

    async def noop(args):
        return "ok"

    tools.register(_tool("noop", noop))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id=f"t{i}", name="noop", input={})],
                stop_reason="tool_use",
            )
            for i in range(5)
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s")
    with pytest.raises(MaxTurnsExceeded, match="max_turns=3"):
        await agent.chat("loop forever", max_turns=3)
    assert agent.messages == []  # fully rolled back


async def test_provider_exception_rolls_back_history():
    tools = ToolRegistry()

    class BoomProvider:
        async def complete(self, *a, **kw):
            raise RuntimeError("network down")

        async def stream(self, *a, **kw):
            raise NotImplementedError

    agent = Agent(provider=BoomProvider(), tools=tools, system="s")
    # Seed some history from a successful prior chat.
    agent.messages.append(Message("user", [TextBlock("earlier")]))
    agent.messages.append(Message("assistant", [TextBlock("ok")]))
    with pytest.raises(RuntimeError, match="network down"):
        await agent.chat("now this fails")
    # The earlier turn survives; the failing chat is gone.
    assert len(agent.messages) == 2
    assert agent.messages[-1].content[0].text == "ok"


async def test_tool_timeout_becomes_error_result():
    tools = ToolRegistry()

    async def slow(args):
        await asyncio.sleep(5)
        return "never"

    tools.register(_tool("slow", slow))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="t1", name="slow", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("recovered")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(
        provider=provider, tools=tools, system="s", tool_timeout=0.05,
    )
    text = await agent.chat("trigger timeout")

    assert text == "recovered"
    err = agent.messages[2].content[0]
    assert isinstance(err, ToolResultBlock)
    assert err.is_error is True
    assert "timed out" in err.content
    assert "slow" in err.content
    assert err.tool_use_id == "t1"
    assert err._tool_name == "slow"


async def test_tool_timeout_isolates_one_slow_call(tmp_path):
    """One slow tool in a parallel batch shouldn't block the fast one's result."""
    tools = ToolRegistry()

    async def fast(args):
        return "fast done"

    async def slow(args):
        await asyncio.sleep(5)
        return "never"

    tools.register(_tool("fast", fast))
    tools.register(_tool("slow", slow))

    provider = MockProvider(
        [
            CompletionResult(
                content=[
                    ToolUseBlock(id="f", name="fast", input={}),
                    ToolUseBlock(id="s", name="slow", input={}),
                ],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("done")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(
        provider=provider, tools=tools, system="s", tool_timeout=0.05,
    )
    await agent.chat("both")
    results = agent.messages[2].content
    assert results[0].content == "fast done"
    assert results[0].is_error is False
    assert results[1].is_error is True
    assert "timed out" in results[1].content


async def test_skills_inject_catalog_and_register_tool(tmp_path):
    (tmp_path / "pdf.md").write_text(
        "---\nname: pdf\ndescription: Handle PDFs\n---\nPDF body here"
    )
    skills = SkillRegistry.from_dir(tmp_path)

    tools = ToolRegistry()
    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="s1", name="load_skill", input={"name": "pdf"})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("got it")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="base prompt", skills=skills)

    assert "SKILL CATALOG" in agent.system
    assert "- pdf: Handle PDFs" in agent.system
    assert "load_skill" in [d.name for d in tools.defs()]

    await agent.chat("read the pdf")
    skill_result = agent.messages[2].content[0]
    assert skill_result.content == "PDF body here"
    assert skill_result._tool_name == "load_skill"


async def test_skills_omitted_leaves_system_untouched():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn")]
    )
    tools = ToolRegistry()
    agent = Agent(provider=provider, tools=tools, system="just this")
    assert agent.system == "just this"
    assert tools.defs() == []


# ---------- SenseGate integration ----------


async def test_sensegate_match_returns_text_unchanged():
    verifier = _ScriptedVerifier([Verdict(matches=True, reason="ok")])
    gate = SenseGate(verifier)
    tools = ToolRegistry()

    async def read(_):
        return "data"

    tools.register(_tool("read_thing", read, kind="read"))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="r1", name="read_thing", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("I read it.")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s", sensegate=gate)
    text = await agent.chat("read please")
    assert text == "I read it."
    assert gate.audit_log == []
    assert len(verifier.calls) == 1


async def test_sensegate_read_mismatch_flags_but_returns():
    verifier = _ScriptedVerifier(
        [Verdict(matches=False, reason="claimed action not invoked")]
    )
    gate = SenseGate(verifier)
    tools = ToolRegistry()

    async def read(_):
        return "data"

    tools.register(_tool("read_thing", read, kind="read"))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="r1", name="read_thing", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("I sent the email!")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s", sensegate=gate)
    text = await agent.chat("read please")
    # Read-only turn — no retry, return as-is, but logged.
    assert text == "I sent the email!"
    assert len(gate.audit_log) == 1
    assert gate.audit_log[0].had_write is False
    # Provider was called only twice (no retry on read mismatch).
    assert len(provider.calls) == 2


async def test_sensegate_write_mismatch_triggers_retry():
    # First verdict: mismatch -> retry. Second: ok.
    verifier = _ScriptedVerifier(
        [
            Verdict(matches=False, reason="claimed success but tool returned error"),
            Verdict(matches=True, reason="now consistent"),
        ]
    )
    gate = SenseGate(verifier)
    tools = ToolRegistry()

    async def write(_):
        return "ack-123"

    tools.register(_tool("send_thing", write, kind="write"))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="w1", name="send_thing", input={})],
                stop_reason="tool_use",
            ),
            # First (lying) final response
            CompletionResult(
                content=[TextBlock("Done — sent successfully (no receipt).")],
                stop_reason="end_turn",
            ),
            # After correction injection: model restates honestly
            CompletionResult(
                content=[TextBlock("Sent. Receipt: ack-123.")],
                stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s", sensegate=gate)
    text = await agent.chat("send it")

    assert text == "Sent. Receipt: ack-123."
    assert len(gate.audit_log) == 1  # the first mismatch was logged
    assert gate.audit_log[0].had_write is True
    # Provider got an extra call due to retry.
    assert len(provider.calls) == 3
    # The correction message should be present in history just before the final assistant msg.
    correction = agent.messages[-2]
    assert correction.role == "user"
    assert "SenseGate" in correction.content[0].text


async def test_sensegate_write_retries_capped_at_max_retries():
    # Always mismatch — should retry exactly max_retries times, then give up.
    verifier = _ScriptedVerifier(
        [Verdict(matches=False, reason="still wrong") for _ in range(5)]
    )
    gate = SenseGate(verifier, max_retries=2)
    tools = ToolRegistry()

    async def write(_):
        return "ok"

    tools.register(_tool("send_thing", write, kind="write"))

    # Provide enough scripted responses for: initial tool turn + 1 final + 2 retries
    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="w1", name="send_thing", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("done v1")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("done v2")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("done v3")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s", sensegate=gate)
    text = await agent.chat("send it")
    # After 2 retries the third assistant text is returned even though it's still flagged.
    assert text == "done v3"
    # Audit logged 3 mismatches (initial + 2 retries).
    assert len(gate.audit_log) == 3
    # Provider called 4 times: 1 tool turn + 3 final-text attempts.
    assert len(provider.calls) == 4


async def test_store_persists_across_agent_lifetimes(tmp_path):
    """Two agents sharing the same store see continuous history."""
    db = tmp_path / "session.db"

    # First agent — fresh start, two chats.
    provider1 = MockProvider(
        [
            CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("yes")], stop_reason="end_turn"),
        ]
    )
    a1 = Agent(
        provider=provider1, tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await a1.load()
    assert a1.messages == []
    await a1.chat("hello")
    await a1.chat("are you there?")
    assert len(a1.messages) == 4

    # Second agent — same store, loads existing history.
    provider2 = MockProvider(
        [CompletionResult(content=[TextBlock("still here")], stop_reason="end_turn")]
    )
    a2 = Agent(
        provider=provider2, tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await a2.load()
    assert len(a2.messages) == 4  # loaded from disk
    await a2.chat("good")
    assert len(a2.messages) == 6

    # Provider on a2 saw the loaded history on its first call.
    sent = provider2.calls[0]["messages"]
    assert len(sent) == 5  # 4 loaded + 1 new "good"
    assert sent[-1].content[0].text == "good"


async def test_store_does_not_persist_on_chat_failure(tmp_path):
    """A chat that raises must not pollute the durable history."""
    db = tmp_path / "session.db"

    class FlakyProvider:
        def __init__(self):
            self.calls = 0

        async def complete(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                return CompletionResult(
                    content=[TextBlock("ok")], stop_reason="end_turn"
                )
            raise RuntimeError("provider blew up")

        async def stream(self, *a, **kw):
            raise NotImplementedError

    agent = Agent(
        provider=FlakyProvider(), tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await agent.load()
    await agent.chat("first")  # succeeds, persisted
    with pytest.raises(RuntimeError, match="blew up"):
        await agent.chat("second")  # fails, must NOT persist

    # Reload — should only see the first successful turn.
    fresh = Agent(
        provider=FlakyProvider(), tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await fresh.load()
    assert len(fresh.messages) == 2  # user(first) + assistant(ok)
    assert fresh.messages[0].content[0].text == "first"


# ---------- compaction / context window ----------


async def test_context_window_keeps_last_n_user_turns():
    """With window=2, provider sees only the last 2 user-text turns."""
    provider = MockProvider(
        [
            CompletionResult(content=[TextBlock("r1")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("r2")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("r3")], stop_reason="end_turn"),
            CompletionResult(content=[TextBlock("r4")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=2,
    )
    await agent.chat("first")
    await agent.chat("second")
    await agent.chat("third")
    await agent.chat("fourth")

    # On the fourth chat, the provider should only see the last 2 user-text
    # turns ("third" + assistant + "fourth") — 3 messages total.
    sent = provider.calls[3]["messages"]
    assert len(sent) == 3
    user_texts = [
        b.text for m in sent if m.role == "user" for b in m.content if isinstance(b, TextBlock)
    ]
    assert user_texts == ["third", "fourth"]

    # The full history is untouched.
    assert len(agent.messages) == 8


async def test_context_window_keeps_tool_chain_intact():
    """A user-text turn includes the assistant tool_use + user tool_result chain."""
    tools = ToolRegistry()

    async def add(args):
        return str(args["a"] + args["b"])

    tools.register(_tool("add", add))

    provider = MockProvider(
        [
            # Turn 1: user "first" -> assistant tool_use -> user tool_result -> assistant text
            CompletionResult(
                content=[ToolUseBlock(id="t1", name="add", input={"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("ans 3")], stop_reason="end_turn"),
            # Turn 2: user "second"
            CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(
        provider=provider, tools=tools, system="s",
        context_window_turns=1,
    )
    await agent.chat("first")
    await agent.chat("second")

    # On the second chat the window is 1, so only "second" + its assistant turn.
    sent = provider.calls[2]["messages"]
    assert all(
        not (m.role == "user" and m.content and isinstance(m.content[0], TextBlock) and m.content[0].text == "first")
        for m in sent
    )
    # And the tool_use/tool_result pair from turn 1 is gone, not orphaned.
    assert all(
        not (m.role == "user" and m.content and isinstance(m.content[0], ToolResultBlock))
        for m in sent
    )


async def test_context_window_does_not_mutate_full_history():
    """The original messages list is never modified by the window slice/clip."""
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=1, max_block_chars=10,
    )
    raw = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 34 chars, will be cut to 10
    await agent.chat(raw)
    # The persisted user message text is intact, not clipped.
    user_text = agent.messages[0].content[0].text
    assert "[clipped]" not in user_text
    assert user_text == raw


async def test_context_window_clips_oversize_blocks_in_view_only():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        max_block_chars=20,
    )
    long_input = "x" * 100
    await agent.chat(long_input)
    sent = provider.calls[0]["messages"]
    sent_text = sent[0].content[0].text
    assert "[clipped]" in sent_text
    assert len(sent_text) < 100
    # But the stored message stays full-length.
    assert agent.messages[0].content[0].text == long_input


async def test_context_window_handles_history_smaller_than_window():
    """If we haven't reached N turns yet, send everything."""
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=10,
    )
    await agent.chat("only one")
    sent = provider.calls[0]["messages"]
    assert len(sent) == 1
    assert sent[0].content[0].text == "only one"


async def test_max_block_chars_none_disables_clipping():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        max_block_chars=None,
    )
    huge = "y" * 50_000
    await agent.chat(huge)
    sent_text = provider.calls[0]["messages"][0].content[0].text
    assert sent_text == huge


async def test_sensegate_disabled_when_omitted():
    tools = ToolRegistry()

    async def write(_):
        return "ok"

    tools.register(_tool("send_thing", write, kind="write"))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="w1", name="send_thing", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(content=[TextBlock("done!")], stop_reason="end_turn"),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s")  # no sensegate
    text = await agent.chat("send it")
    assert text == "done!"
    assert len(provider.calls) == 2
