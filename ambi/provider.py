from typing import AsyncIterator, Protocol

from .types import CompletionResult, Message, ProviderChunk, ToolDef


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> CompletionResult: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> AsyncIterator[ProviderChunk]:
        """Yield TextChunk / ToolCallChunk events, ending with StreamEnd."""
        ...
