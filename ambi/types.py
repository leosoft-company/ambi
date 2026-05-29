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


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A partial text fragment from the provider."""
    text: str


@dataclass
class ToolCallChunk:
    """A complete tool call surfaced during streaming."""
    id: str
    name: str
    input: dict


@dataclass
class StreamEnd:
    """Marks the end of one provider response."""
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
    usage: dict = field(default_factory=dict)


ProviderChunk = TextChunk | ToolCallChunk | StreamEnd


# ---------------------------------------------------------------------------
# Agent-level streaming events (yielded by Agent.chat_stream)
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """Incremental text from the current assistant message."""
    text: str


@dataclass
class ToolUseEvent:
    """The assistant just decided to call a tool."""
    id: str
    name: str
    input: dict


@dataclass
class ToolResultEvent:
    """A tool finished executing."""
    id: str
    name: str
    content: str | list
    is_error: bool


@dataclass
class ToolProgressEvent:
    """A progress message emitted by a long-running tool while it's still in flight."""
    id: str
    name: str
    message: str


@dataclass
class SenseGateFlagEvent:
    """SenseGate detected a mismatch and is about to ask for a retry."""
    reason: str


@dataclass
class ChatComplete:
    """The chat() call has produced its final reply text."""
    final_text: str


AgentEvent = (
    TextDelta
    | ToolUseEvent
    | ToolProgressEvent
    | ToolResultEvent
    | SenseGateFlagEvent
    | ChatComplete
)


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


@dataclass
class CompactionAnchor:
    """One compacted segment of session history.

    Covers messages[from_seq..to_seq] inclusive — those raw messages stay
    on disk for audit/replay; the anchor's summary is what the LLM sees
    for that range in subsequent turns.
    """

    from_seq: int
    to_seq: int
    summary: str
    created_at: str = ""
