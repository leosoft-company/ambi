from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from .types import ToolDef, ToolResultBlock

ToolKind = Literal["read", "write"]


@dataclass
class Tool:
    definition: ToolDef
    handler: Callable[[dict], Awaitable[str | list]]
    kind: ToolKind = "read"


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

    async def invoke(self, name: str, input: dict) -> ToolResultBlock:
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
            result = await tool.handler(input)
            return ToolResultBlock(tool_use_id="", content=result, _tool_name=name)
        except Exception as e:
            return ToolResultBlock(
                tool_use_id="", content=str(e), is_error=True, _tool_name=name
            )
