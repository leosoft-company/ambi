import asyncio

from typing import AsyncIterator

from .provider import LLMProvider
from .sensegate import SenseGate, ToolInvocation, correction_message
from .skills import SkillRegistry, assemble_system, make_load_skill_tool
from .store import SqliteStore
from .tool import ToolRegistry
from .warden import PolicyContext, Warden
from .types import (
    AgentEvent,
    Block,
    ChatComplete,
    CompactionAnchor,
    Message,
    SenseGateFlagEvent,
    StreamEnd,
    TextBlock,
    TextChunk,
    TextDelta,
    ToolCallChunk,
    ToolProgressEvent,
    ToolResultBlock,
    ToolResultEvent,
    ToolUseBlock,
    ToolUseEvent,
)


class MaxTurnsExceeded(RuntimeError):
    """Raised when chat() hits max_turns; session history is rolled back."""

    def __init__(self, max_turns: int):
        super().__init__(
            f"hit max_turns={max_turns}; session history rolled back to before this chat() call"
        )
        self.max_turns = max_turns


class Agent:
    """Stateful agent — `messages` accumulates across `chat()` calls.

    With a `store` and `await agent.load()`, history persists across process
    restarts. Messages are appended to the store after each successful
    `chat()` call. Any chat() that raises is rolled back in memory and
    nothing is persisted.

    Context compaction: only the last `context_window_turns` user-text turns
    are sent to the provider on each call. The full history stays in
    `self.messages` (and on disk if a store is attached) — only the slice
    going to the LLM is trimmed. Individual blocks larger than
    `max_block_chars` are clipped in that slice as well.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        system: str,
        skills: SkillRegistry | None = None,
        tool_timeout: float = 60.0,
        sensegate: SenseGate | None = None,
        store: SqliteStore | None = None,
        session_id: str = "default",
        context_window_turns: int = 5,
        max_block_chars: int | None = 8000,
        compaction_threshold: int = 0,
        warden: Warden | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.skills = skills
        self.tool_timeout = tool_timeout
        self.sensegate = sensegate
        self.store = store
        self.session_id = session_id
        self.context_window_turns = context_window_turns
        self.max_block_chars = max_block_chars
        self.compaction_threshold = compaction_threshold
        self.warden = warden
        if skills is not None:
            tools.register(make_load_skill_tool(skills))
        self.system = assemble_system(system, skills)
        self.messages: list[Message] = []
        self.anchors: list[CompactionAnchor] = []
        self._persisted_count = 0
        self._chat_lock = asyncio.Lock()
        self._compaction_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load persisted messages + compaction anchors. No-op if no store."""
        if self.store is None:
            return
        self.messages = await self.store.load(self.session_id)
        self._persisted_count = len(self.messages)
        self.anchors = await self.store.load_anchors(self.session_id)

    async def chat(self, user_input: str, max_turns: int = 20) -> str:
        async with self._chat_lock:
            text = await self._chat_locked(user_input, max_turns)
        await self._maybe_compact_in_background()
        return text

    async def _chat_locked(self, user_input: str, max_turns: int) -> str:
        snapshot = len(self.messages)
        self.messages.append(Message("user", [TextBlock(user_input)]))

        invocations: list[ToolInvocation] = []
        retries_left = self.sensegate.max_retries if self.sensegate else 0

        try:
            for _ in range(max_turns):
                result = await self.provider.complete(
                    self._context_view(),
                    self.tools.defs(),
                    system=self.system,
                )
                self.messages.append(Message("assistant", result.content))

                if result.stop_reason != "tool_use":
                    final_text = _final_text(result.content)

                    if self.sensegate is not None:
                        verdict = await self.sensegate.check(final_text, invocations)
                        if not verdict.matches:
                            has_write = any(
                                inv.kind == "write" for inv in invocations
                            )
                            if has_write and retries_left > 0:
                                self.messages.append(
                                    correction_message(verdict.reason)
                                )
                                retries_left -= 1
                                continue

                    await self._persist_new()
                    return final_text

                calls = [b for b in result.content if isinstance(b, ToolUseBlock)]
                results = await asyncio.gather(
                    *(self._invoke_with_timeout(c) for c in calls)
                )
                for call, res in zip(calls, results):
                    res.tool_use_id = call.id
                    invocations.append(
                        ToolInvocation(
                            call=call, result=res, kind=self.tools.kind(call.name)
                        )
                    )
                self.messages.append(Message("user", list(results)))
        except BaseException:
            del self.messages[snapshot:]
            raise

        del self.messages[snapshot:]
        raise MaxTurnsExceeded(max_turns)

    async def chat_stream(
        self, user_input: str, max_turns: int = 20
    ) -> AsyncIterator[AgentEvent]:
        """Like chat() but yields events as they happen.

        Yields:
            TextDelta — incremental text from the assistant
            ToolUseEvent — model decided to call a tool
            ToolResultEvent — tool finished executing
            SenseGateFlagEvent — SenseGate flagged a mismatch (retry coming if write)
            ChatComplete — final reply text (always the last event on success)

        Same locking + persistence + rollback guarantees as chat().
        """
        async with self._chat_lock:
            async for ev in self._chat_stream_locked(user_input, max_turns):
                yield ev
        await self._maybe_compact_in_background()

    async def _chat_stream_locked(
        self, user_input: str, max_turns: int
    ) -> AsyncIterator[AgentEvent]:
        snapshot = len(self.messages)
        self.messages.append(Message("user", [TextBlock(user_input)]))

        invocations: list[ToolInvocation] = []
        retries_left = self.sensegate.max_retries if self.sensegate else 0

        try:
            for _ in range(max_turns):
                # Stream one provider turn — accumulate text + tool calls.
                text_buf = ""
                tool_calls: list[ToolCallChunk] = []
                stop_reason = "end_turn"
                async for chunk in self.provider.stream(
                    self._context_view(),
                    self.tools.defs(),
                    system=self.system,
                ):
                    if isinstance(chunk, TextChunk):
                        text_buf += chunk.text
                        yield TextDelta(text=chunk.text)
                    elif isinstance(chunk, ToolCallChunk):
                        tool_calls.append(chunk)
                        yield ToolUseEvent(
                            id=chunk.id, name=chunk.name, input=chunk.input,
                        )
                    elif isinstance(chunk, StreamEnd):
                        stop_reason = chunk.stop_reason

                # Materialize the assistant Message.
                content: list[Block] = []
                if text_buf:
                    content.append(TextBlock(text=text_buf))
                for tc in tool_calls:
                    content.append(
                        ToolUseBlock(id=tc.id, name=tc.name, input=tc.input)
                    )
                self.messages.append(Message("assistant", content))

                if stop_reason != "tool_use":
                    final_text = text_buf

                    if self.sensegate is not None:
                        verdict = await self.sensegate.check(final_text, invocations)
                        if not verdict.matches:
                            has_write = any(
                                inv.kind == "write" for inv in invocations
                            )
                            yield SenseGateFlagEvent(reason=verdict.reason)
                            if has_write and retries_left > 0:
                                self.messages.append(
                                    correction_message(verdict.reason)
                                )
                                retries_left -= 1
                                continue

                    await self._persist_new()
                    yield ChatComplete(final_text=final_text)
                    return

                # Tool calls — invoke them in parallel and forward any
                # progress messages they emit while in flight.
                use_blocks = [b for b in content if isinstance(b, ToolUseBlock)]
                progress_q: asyncio.Queue = asyncio.Queue()

                def _make_progress_cb(call_id: str, call_name: str):
                    async def cb(message: str) -> None:
                        await progress_q.put(
                            ToolProgressEvent(
                                id=call_id, name=call_name, message=message,
                            )
                        )
                    return cb

                tool_tasks = {
                    asyncio.create_task(
                        self._invoke_with_timeout(
                            call,
                            progress=_make_progress_cb(call.id, call.name),
                        )
                    ): call
                    for call in use_blocks
                }
                # Drain progress events while any tool task is still running.
                pending = set(tool_tasks.keys())
                while pending:
                    getter = asyncio.create_task(progress_q.get())
                    done, _ = await asyncio.wait(
                        pending | {getter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if getter in done:
                        yield getter.result()
                    else:
                        getter.cancel()
                        try:
                            await getter
                        except (asyncio.CancelledError, Exception):
                            pass
                    pending -= done

                # Drain any progress events that arrived after the last task
                # completed but before we exited the loop.
                while not progress_q.empty():
                    yield progress_q.get_nowait()

                # Now assemble results in original order.
                results: list[ToolResultBlock] = []
                for call in use_blocks:
                    res = next(
                        (t.result() for t in tool_tasks if tool_tasks[t] is call),
                        None,
                    )
                    assert res is not None  # all tasks are done at this point
                    res.tool_use_id = call.id
                    invocations.append(
                        ToolInvocation(
                            call=call, result=res, kind=self.tools.kind(call.name)
                        )
                    )
                    results.append(res)
                    yield ToolResultEvent(
                        id=call.id,
                        name=call.name,
                        content=res.content,
                        is_error=res.is_error,
                    )
                self.messages.append(Message("user", list(results)))
        except BaseException:
            del self.messages[snapshot:]
            raise

        del self.messages[snapshot:]
        raise MaxTurnsExceeded(max_turns)

    async def _persist_new(self) -> None:
        if self.store is None:
            return
        new = self.messages[self._persisted_count :]
        if not new:
            return
        await self.store.append(new, session_id=self.session_id)
        self._persisted_count = len(self.messages)

    def _context_view(self) -> list[Message]:
        """Return the slice of messages sent to the provider.

        Anchors (if any) get folded in as synthetic user messages summarizing
        their covered range. Verbatim window covers the last N user-text
        turns that aren't covered by an anchor. Long blocks are clipped on a
        copy — original storage is never mutated.
        """
        covered: set[int] = set()
        for a in self.anchors:
            for seq in range(a.from_seq, a.to_seq + 1):
                covered.add(seq)

        cutoff = self._find_window_cutoff(covered)

        out: list[Message] = []
        for a in sorted(self.anchors, key=lambda x: x.from_seq):
            out.append(
                Message(
                    "user",
                    [TextBlock(
                        f"[Earlier conversation, compacted summary of "
                        f"messages {a.from_seq}..{a.to_seq}]: {a.summary}"
                    )],
                )
            )
        for i in range(cutoff, len(self.messages)):
            if i in covered:
                continue
            out.append(self.messages[i])

        if self.max_block_chars is None:
            return out
        return [self._clip_message(m) for m in out]

    def _find_window_cutoff(self, covered: set[int] | None = None) -> int:
        """Index of the Nth-most-recent user-text turn, ignoring covered seqs."""
        target = max(1, self.context_window_turns)
        skip = covered or set()
        count = 0
        for i in range(len(self.messages) - 1, -1, -1):
            if i in skip:
                continue
            m = self.messages[i]
            if (
                m.role == "user"
                and m.content
                and isinstance(m.content[0], TextBlock)
            ):
                count += 1
                if count >= target:
                    return i
        return 0

    def _clip_message(self, m: Message) -> Message:
        return Message(role=m.role, content=[self._clip_block(b) for b in m.content])

    def _clip_block(self, b: Block) -> Block:
        limit = self.max_block_chars
        assert limit is not None
        if isinstance(b, TextBlock) and len(b.text) > limit:
            return TextBlock(text=b.text[:limit] + "\n... [clipped]")
        if (
            isinstance(b, ToolResultBlock)
            and isinstance(b.content, str)
            and len(b.content) > limit
        ):
            return ToolResultBlock(
                tool_use_id=b.tool_use_id,
                content=b.content[:limit] + "\n... [clipped]",
                is_error=b.is_error,
                _tool_name=b._tool_name,
            )
        return b

    # ---------- compaction ----------

    def _next_compaction_range(self) -> tuple[int, int] | None:
        """Return (from_seq, to_seq) for the next batch to compact, or None."""
        if self.compaction_threshold <= 0:
            return None

        covered: set[int] = set()
        for a in self.anchors:
            for seq in range(a.from_seq, a.to_seq + 1):
                covered.add(seq)

        window_cutoff = self._find_window_cutoff(covered)

        # Collect indices that are: not in window (i.e. < cutoff) AND uncovered.
        out_of_window_uncovered: list[int] = [
            i for i in range(window_cutoff)
            if i not in covered
        ]

        # Count user-text turns among them. Trigger when >= threshold.
        user_turn_indices = [
            i for i in out_of_window_uncovered
            if (
                self.messages[i].role == "user"
                and self.messages[i].content
                and isinstance(self.messages[i].content[0], TextBlock)
            )
        ]
        if len(user_turn_indices) < self.compaction_threshold:
            return None

        # Compact a single batch from the oldest uncovered indices forward.
        # Range is contiguous on the storage side: from the first uncovered
        # message to the message just before the (threshold+1)-th user-text
        # turn, so each anchor wraps a clean group of conversational turns.
        from_seq = out_of_window_uncovered[0]
        boundary_user_idx = user_turn_indices[self.compaction_threshold] \
            if len(user_turn_indices) > self.compaction_threshold \
            else None
        to_seq = (boundary_user_idx - 1) if boundary_user_idx is not None \
            else out_of_window_uncovered[-1]
        return from_seq, to_seq

    async def _maybe_compact_in_background(self) -> None:
        if self._next_compaction_range() is None:
            return
        asyncio.create_task(self._compact())

    async def _compact(self) -> None:
        async with self._compaction_lock:
            range_ = self._next_compaction_range()
            if range_ is None:
                return
            from_seq, to_seq = range_
            segment = self.messages[from_seq : to_seq + 1]
            try:
                summary = await self._summarize_segment(segment)
            except Exception:
                return  # leave for next attempt; never crash the chat path
            anchor = CompactionAnchor(
                from_seq=from_seq, to_seq=to_seq, summary=summary,
            )
            self.anchors.append(anchor)
            if self.store is not None:
                try:
                    await self.store.save_anchor(anchor, self.session_id)
                except Exception:
                    pass  # in-memory anchor still works for this process

    async def _summarize_segment(self, segment: list[Message]) -> str:
        from .usage import purpose

        prompt = _build_compaction_prompt(segment)
        with purpose("compaction"):
            result = await self.provider.complete(
                messages=[Message("user", [TextBlock(prompt)])],
                tools=[],
                system=(
                    "You compress conversation segments for long-term agent "
                    "recall. Be terse, factual, third-person. One short "
                    "paragraph; no headers, no bullet lists."
                ),
                max_tokens=512,
            )
        return _final_text(result.content).strip()

    async def _invoke_with_timeout(
        self,
        call: ToolUseBlock,
        progress=None,
    ) -> ToolResultBlock:
        # Pre-execution authorization: ask the Warden whether this call
        # may proceed. Denial becomes an error tool_result so the model
        # sees it and can respond honestly to the user.
        if self.warden is not None:
            decision = await self.warden.authorize(
                PolicyContext(
                    tool_name=call.name,
                    tool_input=call.input,
                    session_id=self.session_id,
                )
            )
            if decision.verdict == "deny":
                return ToolResultBlock(
                    tool_use_id="",
                    content=(
                        f"Denied by policy '{decision.policy_name}': "
                        f"{decision.reason}"
                    ),
                    is_error=True,
                    _tool_name=call.name,
                )

        try:
            return await asyncio.wait_for(
                self.tools.invoke(call.name, call.input, progress=progress),
                timeout=self.tool_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResultBlock(
                tool_use_id="",
                content=f"Tool '{call.name}' timed out after {self.tool_timeout}s",
                is_error=True,
                _tool_name=call.name,
            )


def _final_text(content: list[Block]) -> str:
    return "\n".join(b.text for b in content if isinstance(b, TextBlock))


def _build_compaction_prompt(segment: list[Message]) -> str:
    """Format a segment of session history for the compaction LLM call."""
    lines: list[str] = []
    for m in segment:
        for b in m.content:
            if isinstance(b, TextBlock):
                lines.append(f"{m.role}: {b.text}")
            elif isinstance(b, ToolUseBlock):
                lines.append(f"{m.role}: [tool call] {b.name}({b.input!r})")
            elif isinstance(b, ToolResultBlock):
                tag = "ERROR" if b.is_error else "ok"
                content = b.content if isinstance(b.content, str) else str(b.content)
                lines.append(f"{m.role}: [tool result {b._tool_name} {tag}] {content[:400]}")
    body = "\n".join(lines)
    return (
        "Summarize the following conversation segment as a single short "
        "paragraph for the agent's long-term recall.\n\n"
        "Preserve: stable facts the user shared, decisions or commitments, "
        "tool calls and their outcomes (success/failure, IDs), open threads.\n"
        "Drop: pleasantries, exact phrasing, already-resolved questions.\n\n"
        f"--- segment ---\n{body}\n--- end ---"
    )
