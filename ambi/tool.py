import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from .types import ToolDef, ToolResultBlock

ToolKind = Literal["read", "write"]

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class Tool:
    definition: ToolDef
    handler: Callable[..., Awaitable[str | list]]
    kind: ToolKind = "read"
    _accepts_progress: bool = field(init=False)

    def __post_init__(self) -> None:
        # Cache whether the handler accepts an optional progress callback —
        # signature-inspected once at registration so invocation stays cheap.
        try:
            sig = inspect.signature(self.handler)
            self._accepts_progress = len(sig.parameters) >= 2
        except (TypeError, ValueError):
            self._accepts_progress = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def defs(self) -> list[ToolDef]:
        return [t.definition for t in self._tools.values()]

    def kind(self, name: str) -> ToolKind:
        tool = self._tools.get(name)
        if tool is None:
            # Defensive — if the model invents a tool name the agent loop
            # shouldn't crash. Treat the call as a read.
            return "read"
        return tool.kind

    async def invoke(
        self,
        name: str,
        input: dict,
        progress: ProgressCallback | None = None,
    ) -> ToolResultBlock:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys())) or "(none)"
            return ToolResultBlock(
                tool_use_id="",
                content=f"Tool '{name}' is not registered. Available tools: {available}",
                is_error=True,
                _tool_name=name,
            )
        try:
            if tool._accepts_progress:
                # Always give a callable so handlers don't need to defend
                # against `progress=None` — the chat() (non-streaming) path
                # passes a no-op so progress messages are silently dropped.
                cb = progress if progress is not None else _noop_progress
                result = await tool.handler(input, cb)
            else:
                result = await tool.handler(input)
            return ToolResultBlock(tool_use_id="", content=result, _tool_name=name)
        except Exception as e:
            return ToolResultBlock(
                tool_use_id="", content=str(e), is_error=True, _tool_name=name
            )


async def _noop_progress(_message: str) -> None:
    pass
