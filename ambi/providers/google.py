import os

from google import genai
from google.genai import types as gt

from typing import AsyncIterator

from ..types import (
    Block,
    CompletionResult,
    Message,
    ProviderChunk,
    StreamEnd,
    TextBlock,
    TextChunk,
    ToolCallChunk,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)


class GoogleProvider:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-pro"):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key not provided and neither GEMINI_API_KEY nor "
                "GOOGLE_API_KEY is set. Call ambi.env.load_env() first or "
                "pass api_key explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> CompletionResult:
        contents = [_to_gemini_content(m) for m in messages]
        config = gt.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=(
                [
                    gt.Tool(
                        function_declarations=[
                            gt.FunctionDeclaration(
                                name=t.name,
                                description=t.description,
                                parameters=_schema_to_gemini(t.input_schema),
                            )
                            for t in tools
                        ]
                    )
                ]
                if tools
                else None
            ),
            **provider_kwargs,
        )
        resp = await self.client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        return _from_gemini_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> AsyncIterator[ProviderChunk]:
        """Stream Gemini's response — yields TextChunk / ToolCallChunk and finishes with StreamEnd."""
        contents = [_to_gemini_content(m) for m in messages]
        config = gt.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=(
                [
                    gt.Tool(
                        function_declarations=[
                            gt.FunctionDeclaration(
                                name=t.name,
                                description=t.description,
                                parameters=_schema_to_gemini(t.input_schema),
                            )
                            for t in tools
                        ]
                    )
                ]
                if tools
                else None
            ),
            **provider_kwargs,
        )

        stop_reason = "end_turn"
        usage: dict = {}
        tool_call_index = 0
        had_tool_call = False

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config,
        ):
            cand = chunk.candidates[0] if chunk.candidates else None
            if cand is None:
                continue
            for part in cand.content.parts or []:
                text = getattr(part, "text", None)
                if text:
                    yield TextChunk(text=text)
                    continue
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    had_tool_call = True
                    yield ToolCallChunk(
                        id=f"{fc.name}:{tool_call_index}",
                        name=fc.name,
                        input=dict(fc.args or {}),
                    )
                    tool_call_index += 1

            if cand.finish_reason is not None:
                fr_name = getattr(cand.finish_reason, "name", str(cand.finish_reason))
                if not had_tool_call:
                    stop_reason = {
                        "STOP": "end_turn",
                        "MAX_TOKENS": "max_tokens",
                        "STOP_SEQUENCE": "stop_sequence",
                    }.get(fr_name, "end_turn")

            um = getattr(chunk, "usage_metadata", None)
            if um is not None:
                usage = {
                    "input_tokens": getattr(um, "prompt_token_count", 0),
                    "output_tokens": getattr(um, "candidates_token_count", 0),
                }

        if had_tool_call:
            stop_reason = "tool_use"
        yield StreamEnd(stop_reason=stop_reason, usage=usage)


def _to_gemini_content(msg: Message) -> gt.Content:
    role = "model" if msg.role == "assistant" else "user"
    parts: list[gt.Part] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(gt.Part(text=block.text))
        elif isinstance(block, ToolUseBlock):
            parts.append(
                gt.Part(
                    function_call=gt.FunctionCall(name=block.name, args=block.input)
                )
            )
        elif isinstance(block, ToolResultBlock):
            response = (
                {"error": block.content} if block.is_error else {"result": block.content}
            )
            parts.append(
                gt.Part(
                    function_response=gt.FunctionResponse(
                        name=block._tool_name, response=response
                    )
                )
            )
    return gt.Content(role=role, parts=parts)


def _from_gemini_response(resp) -> CompletionResult:
    cand = resp.candidates[0]
    blocks: list[Block] = []
    has_tool_call = False
    for i, part in enumerate(cand.content.parts):
        if getattr(part, "text", None):
            blocks.append(TextBlock(text=part.text))
        elif getattr(part, "function_call", None):
            has_tool_call = True
            fc = part.function_call
            blocks.append(
                ToolUseBlock(
                    id=f"{fc.name}:{i}",
                    name=fc.name,
                    input=dict(fc.args or {}),
                )
            )

    if has_tool_call:
        stop = "tool_use"
    else:
        # Compare by name so we tolerate SDK changes (some enum values
        # come and go between google-genai releases).
        fr_name = getattr(cand.finish_reason, "name", str(cand.finish_reason))
        stop = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "STOP_SEQUENCE": "stop_sequence",
        }.get(fr_name, "end_turn")

    usage = getattr(resp, "usage_metadata", None)
    return CompletionResult(
        content=blocks,
        stop_reason=stop,
        usage={
            "input_tokens": getattr(usage, "prompt_token_count", 0),
            "output_tokens": getattr(usage, "candidates_token_count", 0),
        },
    )


_TYPE_MAP = {
    "string": gt.Type.STRING,
    "number": gt.Type.NUMBER,
    "integer": gt.Type.INTEGER,
    "boolean": gt.Type.BOOLEAN,
    "array": gt.Type.ARRAY,
    "object": gt.Type.OBJECT,
}

_STRIP_KEYS = {"$schema", "additionalProperties", "$id", "$ref", "definitions"}


def _schema_to_gemini(schema: dict) -> gt.Schema:
    """Translate a JSON Schema dict into gt.Schema, dropping unsupported keys."""
    cleaned = {k: v for k, v in schema.items() if k not in _STRIP_KEYS}
    kwargs: dict = {}
    if "type" in cleaned:
        kwargs["type"] = _TYPE_MAP.get(cleaned["type"], gt.Type.STRING)
    if "description" in cleaned:
        kwargs["description"] = cleaned["description"]
    if "enum" in cleaned:
        kwargs["enum"] = [str(v) for v in cleaned["enum"]]
    if "required" in cleaned:
        kwargs["required"] = list(cleaned["required"])
    if "properties" in cleaned:
        kwargs["properties"] = {
            name: _schema_to_gemini(sub) for name, sub in cleaned["properties"].items()
        }
    if "items" in cleaned:
        kwargs["items"] = _schema_to_gemini(cleaned["items"])
    return gt.Schema(**kwargs)
