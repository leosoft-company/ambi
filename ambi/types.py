from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list
    is_error: bool = False
    _tool_name: str = ""


Block = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[Block]


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict


@dataclass
class CompletionResult:
    content: list[Block]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    usage: dict = field(default_factory=dict)
