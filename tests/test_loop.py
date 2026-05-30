import asyncio

import pytest

from ambi.loop import Agent, MaxTurnsExceeded
from ambi.sensegate import SenseGate, Verdict
from ambi.skills import SkillRegistry
from ambi.store import SqliteStore
from ambi.tool import Tool, ToolKind, ToolRegistry
from ambi.types import (
    ChatComplete,
    CompactionAnchor,
    CompletionResult,
    Message,
    StreamEnd,
    TextBlock,
    TextChunk,
    TextDelta,
    ToolCallChunk,
    ToolDef,
    ToolResultBlock,
    ToolResultEvent,
    ToolUseBlock,
    ToolUseEvent,
)

from tests.mock_provider import MockProvider, MockStreamProvider


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
            raise RuntimeError("network down")
            yield  # unreachable — makes this an async generator

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
    # The correction (retry) turn must be tool-free so it can't re-invoke a
    # write — the first two turns saw tools, the correction turn saw none.
    assert provider.calls[0]["tools"] != []
    assert provider.calls[2]["tools"] == []


async def test_sensegate_retry_cannot_reexecute_write():
    """A retry must not re-run a side-effecting write that already fired."""
    verifier = _ScriptedVerifier(
        [
            Verdict(matches=False, reason="claimed success, verify"),
            Verdict(matches=True, reason="ok"),
        ]
    )
    gate = SenseGate(verifier)
    tools = ToolRegistry()
    sends: list[int] = []

    async def send(_):
        sends.append(1)
        return "sent"

    tools.register(_tool("send_thing", send, kind="write"))

    provider = MockProvider(
        [
            CompletionResult(
                content=[ToolUseBlock(id="w1", name="send_thing", input={})],
                stop_reason="tool_use",
            ),
            CompletionResult(
                content=[TextBlock("Sent it!")], stop_reason="end_turn",
            ),
            # If the retry turn were given tools, a model could re-issue the
            # send here. With tools withheld it can only restate.
            CompletionResult(
                content=[TextBlock("Sent — receipt: sent.")], stop_reason="end_turn",
            ),
        ]
    )
    agent = Agent(provider=provider, tools=tools, system="s", sensegate=gate)
    await agent.chat("send it")

    assert sends == [1]  # executed exactly once despite the retry


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
            # Mirror complete(): first turn succeeds, later turns blow up.
            self.calls += 1
            if self.calls == 1:
                yield TextChunk(text="ok")
                yield StreamEnd(stop_reason="end_turn")
                return
            raise RuntimeError("provider blew up")

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


# ---------- streaming ----------


async def test_chat_stream_emits_text_deltas_and_completes():
    provider = MockStreamProvider([
        [
            TextChunk(text="hello "),
            TextChunk(text="there"),
            StreamEnd(stop_reason="end_turn"),
        ],
    ])
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s")
    events = [ev async for ev in agent.chat_stream("hi")]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["hello ", "there"]
    complete = [e for e in events if isinstance(e, ChatComplete)]
    assert complete and complete[0].final_text == "hello there"


async def test_chat_stream_yields_tool_use_and_result_events():
    tools = ToolRegistry()

    async def add(args):
        return str(args["a"] + args["b"])

    tools.register(_tool("add", add))

    provider = MockStreamProvider([
        # Turn 1: tool use chunk + stream end
        [
            ToolCallChunk(id="t1", name="add", input={"a": 1, "b": 2}),
            StreamEnd(stop_reason="tool_use"),
        ],
        # Turn 2: final text
        [
            TextChunk(text="3"),
            StreamEnd(stop_reason="end_turn"),
        ],
    ])
    agent = Agent(provider=provider, tools=tools, system="s")
    events = [ev async for ev in agent.chat_stream("add 1+2")]
    tool_uses = [e for e in events if isinstance(e, ToolUseEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    completes = [e for e in events if isinstance(e, ChatComplete)]
    assert tool_uses and tool_uses[0].name == "add"
    assert tool_results and tool_results[0].content == "3"
    assert completes and completes[0].final_text == "3"


async def test_chat_stream_persists_on_success(tmp_path):
    from ambi.store import SqliteStore

    db = tmp_path / "session.db"
    provider = MockStreamProvider([
        [TextChunk(text="ok"), StreamEnd(stop_reason="end_turn")],
    ])
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await agent.load()
    _ = [ev async for ev in agent.chat_stream("first")]

    fresh = Agent(
        provider=MockStreamProvider([]), tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
    )
    await fresh.load()
    assert len(fresh.messages) == 2


# ---------- compaction ----------


async def test_compaction_disabled_by_default():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s")
    assert agent.compaction_threshold == 0
    assert agent._next_compaction_range() is None


async def test_compaction_triggers_when_threshold_reached():
    """With threshold=3 and window=2, an extra 4 turns past the window should fire."""
    provider = MockProvider(
        [
            CompletionResult(content=[TextBlock("r")], stop_reason="end_turn")
            for _ in range(6)
        ]
        + [CompletionResult(content=[TextBlock("summary text")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=2,
        compaction_threshold=3,
    )
    for i in range(6):
        await agent.chat(f"turn {i}")
    # Compaction is fire-and-forget; await it once.
    await agent._compact()
    assert len(agent.anchors) == 1
    a = agent.anchors[0]
    # Should cover at least the first 3 user-text turns (and the assistant
    # messages between them).
    assert a.from_seq == 0
    assert a.to_seq >= 5  # 3 user + 3 assistant minimum


async def test_compaction_skips_empty_summary_no_history_drop():
    """If the summarizer returns blank, no anchor is created — the messages
    stay verbatim instead of being hidden behind an empty summary."""
    # 6 turns of normal replies, then a BLANK summary response for _compact.
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("r")], stop_reason="end_turn") for _ in range(6)]
        + [CompletionResult(content=[TextBlock("   ")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=2, compaction_threshold=3,
    )
    for i in range(6):
        await agent.chat(f"turn {i}")
    await agent._compact()

    assert agent.anchors == []  # blank summary → no anchor
    # Nothing is hidden: every stored message is still reachable in the view
    # (no covered range), so old turns aren't silently dropped.
    view = agent._context_view()
    assert not any("compacted summary" in b.text
                   for m in view for b in m.content if isinstance(b, TextBlock))


async def test_context_view_ignores_blank_summary_anchor():
    """A blank-summary anchor on disk must not hide its messages."""
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    # Wide window so all turns would be in view — the only thing that could
    # hide u0/a0 is the anchor, which we want ignored.
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=10, compaction_threshold=0,
    )
    for i in range(3):
        agent.messages.append(Message("user", [TextBlock(f"u{i}")]))
        agent.messages.append(Message("assistant", [TextBlock(f"a{i}")]))
    # Corrupt/legacy anchor with an empty summary covering u0..a1.
    agent.anchors.append(CompactionAnchor(from_seq=0, to_seq=3, summary="  "))

    view = agent._context_view()
    texts = [b.text for m in view for b in m.content if isinstance(b, TextBlock)]
    # The blank anchor is ignored — no synthetic summary line…
    assert not any("compacted summary" in t for t in texts)
    # …and the messages it claimed to cover are not forced out of view by it.
    assert "u0" in texts and "a0" in texts


async def test_next_compaction_range_none_below_threshold():
    """Exactly threshold-1 eligible turns must not trigger compaction."""
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("r")], stop_reason="end_turn") for _ in range(3)]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=1, compaction_threshold=3,
    )
    for i in range(3):
        await agent.chat(f"t{i}")
    # 3 turns, window keeps 1, so only 2 are out-of-window — below threshold 3.
    assert agent._next_compaction_range() is None


async def test_context_view_folds_anchor_in():
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")]
    )
    agent = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        context_window_turns=2,
        compaction_threshold=0,  # we'll inject the anchor manually
    )
    # 8 messages: 4 user, 4 assistant — but we'll insert an anchor over 0..3
    for i in range(4):
        agent.messages.append(Message("user", [TextBlock(f"u{i}")]))
        agent.messages.append(Message("assistant", [TextBlock(f"a{i}")]))
    agent.anchors.append(
        CompactionAnchor(from_seq=0, to_seq=3, summary="early bits")
    )

    view = agent._context_view()
    # First message in view: the synthetic anchor summary
    first = view[0]
    assert "early bits" in first.content[0].text
    # Then the verbatim window of the last 2 user-text turns (u2, a2, u3, a3)
    rest_texts = [
        m.content[0].text for m in view[1:]
        if isinstance(m.content[0], TextBlock)
    ]
    assert "u2" in rest_texts
    assert "u3" in rest_texts
    # The covered messages u0, u1 should NOT appear verbatim
    assert "u0" not in rest_texts
    assert "u1" not in rest_texts
    # Original messages are untouched (non-destructive)
    assert len(agent.messages) == 8


async def test_anchor_persists_to_store_and_reloads(tmp_path):
    from ambi.store import SqliteStore

    db = tmp_path / "session.db"
    provider = MockProvider(
        [CompletionResult(content=[TextBlock("ok")], stop_reason="end_turn")] * 5
        + [CompletionResult(content=[TextBlock("summary")], stop_reason="end_turn")]
    )
    a1 = Agent(
        provider=provider, tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
        context_window_turns=1,
        compaction_threshold=2,
    )
    await a1.load()
    for i in range(3):
        await a1.chat(f"t{i}")
    await a1._compact()
    assert len(a1.anchors) == 1
    original_anchor = a1.anchors[0]

    # Fresh agent reads the persisted anchor from the store.
    a2 = Agent(
        provider=MockProvider([]), tools=ToolRegistry(), system="s",
        store=SqliteStore(db),
        context_window_turns=1,
        compaction_threshold=2,
    )
    await a2.load()
    assert len(a2.anchors) == 1
    assert a2.anchors[0].from_seq == original_anchor.from_seq
    assert a2.anchors[0].to_seq == original_anchor.to_seq
    assert a2.anchors[0].summary == original_anchor.summary


# ---------- streaming tool progress ----------


async def test_progress_aware_tool_yields_progress_events():
    """A handler with a (input, progress) signature gets a callback and its
    progress messages surface as ToolProgressEvent in chat_stream."""
    tools = ToolRegistry()

    async def slow_with_progress(args, progress):
        await progress("starting…")
        await progress("halfway…")
        await progress("done")
        return "final result"

    tools.register(Tool(
        definition=ToolDef(
            name="slow",
            description="slow tool",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        handler=slow_with_progress,
    ))

    provider = MockStreamProvider([
        [
            ToolCallChunk(id="t1", name="slow", input={}),
            StreamEnd(stop_reason="tool_use"),
        ],
        [
            TextChunk(text="ok"),
            StreamEnd(stop_reason="end_turn"),
        ],
    ])

    agent = Agent(provider=provider, tools=tools, system="s")
    events = [ev async for ev in agent.chat_stream("go")]

    from ambi.types import ToolProgressEvent
    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert [p.message for p in progress] == ["starting…", "halfway…", "done"]
    assert all(p.id == "t1" and p.name == "slow" for p in progress)

    # ToolResultEvent comes after all progress events
    progress_indices = [i for i, e in enumerate(events) if isinstance(e, ToolProgressEvent)]
    result_index = next(i for i, e in enumerate(events) if isinstance(e, ToolResultEvent))
    assert max(progress_indices) < result_index

    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.content == "final result"


async def test_legacy_tool_works_without_progress_arg():
    """Handlers with the single-arg signature still work, no progress events emitted."""
    tools = ToolRegistry()

    async def plain(args):
        return "done"

    tools.register(_tool("plain", plain))

    provider = MockStreamProvider([
        [
            ToolCallChunk(id="t1", name="plain", input={}),
            StreamEnd(stop_reason="tool_use"),
        ],
        [TextChunk(text="ok"), StreamEnd(stop_reason="end_turn")],
    ])
    agent = Agent(provider=provider, tools=tools, system="s")
    events = [ev async for ev in agent.chat_stream("go")]

    from ambi.types import ToolProgressEvent
    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert progress == []


async def test_chat_path_ignores_progress_callbacks():
    """Non-streaming chat() doesn't need progress; tools that emit it still work."""
    tools = ToolRegistry()

    async def with_progress(args, progress):
        await progress("noise")
        return "ok"

    tools.register(Tool(
        definition=ToolDef(
            name="x",
            description="x",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        handler=with_progress,
    ))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(id="t1", name="x", input={})],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("done")], stop_reason="end_turn"),
    ])
    agent = Agent(provider=provider, tools=tools, system="s")
    text = await agent.chat("go")
    assert text == "done"
    # The result block should have the tool's final return value.
    res = agent.messages[2].content[0]
    assert res.content == "ok"


# ---------- Warden integration ----------


async def test_warden_denial_becomes_error_tool_result():
    """A Warden deny verdict should produce an error ToolResultBlock that
    the agent can see and respond to, without invoking the handler."""
    from ambi.warden import ArgvValidatorPolicy, Warden

    tools = ToolRegistry()
    invocations = []

    async def fake_run(args):
        invocations.append(args)
        return "ran"

    tools.register(_tool("run_command", fake_run, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command",
                input={"argv": ["git", "push", "--force"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(
            content=[TextBlock("can't do that")],
            stop_reason="end_turn",
        ),
    ])
    warden = Warden(policies=[ArgvValidatorPolicy(forbid=["push --force"])])
    agent = Agent(
        provider=provider, tools=tools, system="s", warden=warden,
    )
    text = await agent.chat("force push please")

    assert text == "can't do that"
    # Handler must NOT have run.
    assert invocations == []
    # The recorded tool_result should be the policy denial.
    res = agent.messages[2].content[0]
    assert isinstance(res, ToolResultBlock)
    assert res.is_error
    assert "Denied by policy" in res.content
    assert "argv_validator" in res.content


async def test_require_confirmation_fails_closed_without_confirmer():
    """A require_confirmation verdict with no confirmer must NOT execute."""
    from ambi.warden import RequireConfirmationPolicy, Warden

    tools = ToolRegistry()
    ran = []

    async def push(args):
        ran.append(args)
        return "pushed"

    tools.register(_tool("run_command", push, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command",
                input={"argv": ["git", "push", "origin", "main"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("blocked")], stop_reason="end_turn"),
    ])
    warden = Warden(policies=[RequireConfirmationPolicy(argv_patterns=["git push"])])
    agent = Agent(provider=provider, tools=tools, system="s", warden=warden)
    await agent.chat("push it")

    assert ran == []  # never executed
    res = agent.messages[2].content[0]
    assert isinstance(res, ToolResultBlock)
    assert res.is_error
    assert "Requires confirmation" in res.content
    assert "fail-closed" in res.content


async def test_require_confirmation_executes_when_approved():
    from ambi.warden import RequireConfirmationPolicy, Warden

    tools = ToolRegistry()
    ran = []

    async def push(args):
        ran.append(args)
        return "pushed"

    tools.register(_tool("run_command", push, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command",
                input={"argv": ["git", "push"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("done")], stop_reason="end_turn"),
    ])
    warden = Warden(policies=[RequireConfirmationPolicy(argv_patterns=["git push"])])
    seen = []

    async def approve(ctx, decision):
        seen.append((ctx.tool_name, decision.verdict))
        return True

    agent = Agent(
        provider=provider, tools=tools, system="s",
        warden=warden, confirm=approve,
    )
    await agent.chat("push it")

    assert len(ran) == 1  # executed after approval
    assert seen == [("run_command", "require_confirmation")]
    assert agent.messages[2].content[0].content == "pushed"


async def test_require_confirmation_declined_does_not_execute():
    from ambi.warden import RequireConfirmationPolicy, Warden

    tools = ToolRegistry()
    ran = []

    async def push(args):
        ran.append(args)
        return "pushed"

    tools.register(_tool("run_command", push, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command", input={"argv": ["git", "push"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("ok, skipped")], stop_reason="end_turn"),
    ])
    warden = Warden(policies=[RequireConfirmationPolicy(argv_patterns=["git push"])])

    async def decline(ctx, decision):
        return False

    agent = Agent(
        provider=provider, tools=tools, system="s",
        warden=warden, confirm=decline,
    )
    await agent.chat("push it")

    assert ran == []
    res = agent.messages[2].content[0]
    assert res.is_error
    assert "declined by user" in res.content


async def test_confirmer_error_fails_closed():
    """A confirmer that raises must be read as a decline, not approval."""
    from ambi.warden import RequireConfirmationPolicy, Warden

    tools = ToolRegistry()
    ran = []

    async def push(args):
        ran.append(args)
        return "pushed"

    tools.register(_tool("run_command", push, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command", input={"argv": ["git", "push"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("blocked")], stop_reason="end_turn"),
    ])
    warden = Warden(policies=[RequireConfirmationPolicy(argv_patterns=["git push"])])

    async def boom(ctx, decision):
        raise RuntimeError("confirmer crashed")

    agent = Agent(
        provider=provider, tools=tools, system="s",
        warden=warden, confirm=boom,
    )
    await agent.chat("push it")
    assert ran == []
    assert agent.messages[2].content[0].is_error


async def test_warden_audit_log_records_each_authorization():
    from ambi.warden import ArgvValidatorPolicy, Warden

    tools = ToolRegistry()

    async def noop(args):
        return "ok"

    tools.register(_tool("run_command", noop, kind="write"))

    provider = MockProvider([
        CompletionResult(
            content=[ToolUseBlock(
                id="t1", name="run_command",
                input={"argv": ["ls"]},
            )],
            stop_reason="tool_use",
        ),
        CompletionResult(content=[TextBlock("done")], stop_reason="end_turn"),
    ])
    warden = Warden(policies=[ArgvValidatorPolicy(forbid=["rm -rf"])])
    agent = Agent(
        provider=provider, tools=tools, system="s", warden=warden,
    )
    await agent.chat("ls please")
    assert len(warden.audit_log) == 1
    assert warden.audit_log[0].verdict == "allow"
    assert warden.audit_log[0].tool_name == "run_command"
