from typing import AsyncIterator, Protocol

from .types import Block, CompletionResult, Message, ToolDef


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> CompletionResult: ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> AsyncIterator[Block]: ...
