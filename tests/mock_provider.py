"""Shared test fixtures — MockProvider for offline loop testing."""

from __future__ import annotations

from typing import Any

from ambi.types import (
    CompletionResult,
    Message,
    StreamEnd,
    TextBlock,
    TextChunk,
    ToolCallChunk,
    ToolDef,
    ToolUseBlock,
)


class MockProvider:
    """Returns scripted CompletionResults in order; records every call.

    Used to drive Agent.run() through prescribed turns without hitting an LLM.
    """

    def __init__(self, responses: list[CompletionResult]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs: Any,
    ) -> CompletionResult:
        # Snapshot the messages list — the caller (Agent) mutates the same
        # list across turns, so tests need to see what was sent *at this call*.
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "system": system,
                "max_tokens": max_tokens,
                "kwargs": provider_kwargs,
            }
        )
        if not self.responses:
            raise RuntimeError("MockProvider out of scripted responses")
        return self.responses.pop(0)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs: Any,
    ):
        # Synthesize a chunk stream from the next scripted CompletionResult so
        # the same scripts drive both the complete() and stream() paths. Calls
        # are recorded identically, so call-counting assertions are unaffected
        # whether the agent goes through chat() (now streaming) or chat_stream().
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "system": system,
                "max_tokens": max_tokens,
                "kwargs": provider_kwargs,
            }
        )
        if not self.responses:
            raise RuntimeError("MockProvider out of scripted responses")
        result = self.responses.pop(0)
        for block in result.content:
            if isinstance(block, TextBlock):
                yield TextChunk(text=block.text)
            elif isinstance(block, ToolUseBlock):
                yield ToolCallChunk(id=block.id, name=block.name, input=block.input)
        yield StreamEnd(stop_reason=result.stop_reason, usage=result.usage)


class MockStreamProvider:
    """Streams scripted chunks. `scripts` is a list of lists of chunks —
    one inner list per provider turn.
    """

    def __init__(self, scripts: list[list]):
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("MockStreamProvider only implements stream()")

    async def stream(
        self,
        messages,
        tools,
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs: Any,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "system": system,
                "max_tokens": max_tokens,
                "kwargs": provider_kwargs,
            }
        )
        if not self.scripts:
            raise RuntimeError("MockStreamProvider out of scripts")
        for chunk in self.scripts.pop(0):
            yield chunk
