import asyncio

from typing import AsyncIterator

from .provider import LLMProvider
from .sensegate import SenseGate, ToolInvocation, correction_message
from .skills import SkillRegistry, assemble_system, make_load_skill_tool
from .store import SqliteStore
from .tool import ToolRegistry
from .types import (
    AgentEvent,
    Block,
    ChatComplete,
    Message,
    SenseGateFlagEvent,
    StreamEnd,
    TextBlock,
    TextChunk,
    TextDelta,
    ToolCallChunk,
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
        if skills is not None:
            tools.register(make_load_skill_tool(skills))
        self.system = assemble_system(system, skills)
        self.messages: list[Message] = []
        self._persisted_count = 0
        self._chat_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load persisted messages from the store. No-op if no store."""
        if self.store is None:
            return
        self.messages = await self.store.load(self.session_id)
        self._persisted_count = len(self.messages)

    async def chat(self, user_input: str, max_turns: int = 20) -> str:
        async with self._chat_lock:
            return await self._chat_locked(user_input, max_turns)

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

                # Tool calls — invoke them in parallel.
                use_blocks = [b for b in content if isinstance(b, ToolUseBlock)]
                results = await asyncio.gather(
                    *(self._invoke_with_timeout(c) for c in use_blocks)
                )
                for call, res in zip(use_blocks, results):
                    res.tool_use_id = call.id
                    invocations.append(
                        ToolInvocation(
                            call=call, result=res, kind=self.tools.kind(call.name)
                        )
                    )
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
        """Return the trimmed/clipped slice of messages sent to the provider.

        Walks backward from the end of self.messages, counting "user-text"
        turns (user messages whose first block is a TextBlock — i.e. real
        inputs, not tool_result wrappers). Slices from the Nth such turn
        forward, then clips any oversize block in the resulting copy.
        """
        cutoff = self._find_window_cutoff()
        window = self.messages[cutoff:]
        if self.max_block_chars is None:
            return window
        return [self._clip_message(m) for m in window]

    def _find_window_cutoff(self) -> int:
        target = max(1, self.context_window_turns)
        count = 0
        for i in range(len(self.messages) - 1, -1, -1):
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

    async def _invoke_with_timeout(self, call: ToolUseBlock) -> ToolResultBlock:
        try:
            return await asyncio.wait_for(
                self.tools.invoke(call.name, call.input),
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
