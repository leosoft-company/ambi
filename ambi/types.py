import re
import unicodedata
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


# ---------------------------------------------------------------------------
# Prompt-injection defense
# ---------------------------------------------------------------------------
#
# Tool outputs are UNTRUSTED. A fetched web page, an email body, or a
# compromised MCP server can embed text that tries to hijack the agent
# ("ignore previous instructions, you are now…"). Two layers of defense:
#
#   1. sanitize_tool_output() neutralizes the common *obfuscation* tricks —
#      invisible/zero-width chars and Unicode "tag" smuggling, bidi overrides,
#      and compatibility forms (fullwidth/ligatures) — then strips the most
#      common plaintext injection markers.
#   2. wrap_tool_output() wraps every result in an explicit trust envelope
#      so the model treats the enclosed text as data to reason about, never
#      as commands to obey (see cli/system.md for the matching framing).
#
# LIMITS — string matching is the weakest layer and is not the real control.
# It cannot see through *content* encodings (base64/hex/rot13) or cross-script
# homoglyphs (Cyrillic 'а' for Latin 'a'): an attacker who base64-encodes
# "ignore previous instructions" defeats every pattern here, because the model
# is the thing that would decode and act on it. The durable defenses are
# encoding-agnostic and live elsewhere: the trust envelope + system framing
# (treat envelope contents as data regardless of how they're encoded), and the
# structural controls — Warden authorizing every write, SenseGate verifying
# claims, human-in-the-loop on outbound actions. Treat this function as
# defense-in-depth noise reduction for the naive 90%, never as a gate.

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    # "ignore / disregard / forget [all] [previous] instructions/prompts/context"
    re.compile(
        r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}?"
        r"\b(?:previous|prior|above|preceding|earlier|all|the)\b[^.\n]{0,40}?"
        r"\b(?:instruction|instructions|prompt|prompts|message|messages|"
        r"context|rule|rules|direction|directions)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\byou are now\b(?:\s+(?:a|an|the))?", re.IGNORECASE),
    re.compile(r"\byou are no longer\b", re.IGNORECASE),
    re.compile(r"\bnew\s+(?:instructions|prompt|system\s+prompt)\s*:", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\s*:", re.IGNORECASE),
    re.compile(
        r"\boverride\b[^.\n]{0,30}?"
        r"\b(?:instruction|instructions|rule|rules|guideline|guidelines)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpretend\b[^.\n]{0,20}?\byou are\b", re.IGNORECASE),
    # Defeat attempts to forge or break out of the trust envelope / system role.
    re.compile(r"<\s*/?\s*tool_output\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\b[^>]*>", re.IGNORECASE),
)

_REDACTION = "[redacted]"

# Invisible / formatting chars stripped before matching: zero-width spaces,
# joiners, BOM, soft hyphen. They render as nothing to a human but let an
# attacker split a marker ("ig<ZWSP>nore") past the patterns. Stripped quietly
# — these also appear in benign text (BOMs, emoji ZWJ), so removal alone is not
# treated as suspicious.
_INVISIBLE_RE = re.compile(
    "[­"          # soft hyphen
    "​-‏"    # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "⁠-⁤"    # word joiner + invisible math operators
    "﻿]"          # BOM / zero-width no-break space
)

# High-signal smuggling chars with essentially no benign use in tool *text*:
# bidi overrides/isolates (visual reordering attacks) and the Unicode tags
# block (U+E0000–E007F) used for "ASCII smuggling" of invisible instructions.
# Presence alone flags the result as suspicious.
_SMUGGLING_RE = re.compile(
    "[‪-‮"   # bidi embeddings + overrides (LRE/RLE/PDF/LRO/RLO)
    "⁦-⁯"    # bidi isolates + deprecated format chars
    "\U000e0000-\U000e007f]"  # Unicode tags block (ASCII smuggling)
)


def sanitize_tool_output(text: str) -> tuple[str, bool]:
    """Neutralize obfuscation and strip known injection markers from untrusted
    tool output.

    Returns ``(cleaned_text, found)``. ``found`` is True when something
    genuinely suspicious was seen — a plaintext marker redacted, or a
    smuggling char (bidi override / Unicode tag) present — so callers can flag
    the result. Quiet cleanup (zero-width removal, NFKC folding) does not by
    itself set the flag, to avoid crying wolf on ligatures and BOMs.

    Conservative by design, and blind to content encodings and homoglyphs —
    see the module note. The trust envelope, not this function, is the control.
    """
    suspicious = bool(_SMUGGLING_RE.search(text))

    # Drop invisible chars, then collapse compatibility forms so fullwidth
    # ("ｉｇｎｏｒｅ") and ligatures can't hide a marker from the patterns below.
    cleaned = _INVISIBLE_RE.sub("", text)
    cleaned = _SMUGGLING_RE.sub("", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    found = suspicious
    for pat in _INJECTION_PATTERNS:
        cleaned, n = pat.subn(_REDACTION, cleaned)
        if n:
            found = True
    return cleaned, found


def wrap_tool_output(
    content: str, *, trust: str = "data", sanitized: bool = False
) -> str:
    """Wrap tool output in an explicit trust envelope.

    The envelope tells the model the enclosed text is *data to reason about*,
    never instructions to follow. ``sanitized=True`` is surfaced as an
    attribute so the model knows an injection marker was stripped from this
    result.
    """
    attrs = f' trust="{trust}"'
    if sanitized:
        attrs += ' sanitized="true"'
    return f"<tool_output{attrs}>\n{content}\n</tool_output>"
