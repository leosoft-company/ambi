"""Live smoke tests against real Gemini — proves the adapter + loop work end-to-end.

Run all: `uv run pytest`
Skip these: `uv run pytest -m "not smoke"`
Only these: `uv run pytest -m smoke`
"""

import os

import pytest

from ambi.loop import Agent
from ambi.providers.google import GoogleProvider
from ambi.skills import SkillDef, SkillRegistry
from ambi.tool import Tool, ToolRegistry
from ambi.types import TextBlock, ToolDef, ToolUseBlock

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    ),
]

MODEL = "gemini-2.5-flash"


def _all_assistant_text(messages) -> str:
    return " ".join(
        b.text for m in messages if m.role == "assistant"
        for b in m.content if isinstance(b, TextBlock)
    ).lower()


async def test_simple_completion():
    """Adapter handles text-only round trip."""
    provider = GoogleProvider(model=MODEL)
    agent = Agent(
        provider=provider,
        tools=ToolRegistry(),
        system="Reply with exactly one word.",
    )
    text = await agent.chat("Reply with the word 'banana' and nothing else.")
    assert "banana" in text.lower()


async def test_tool_call_round_trip():
    """Model calls a tool, gets a result, produces a final answer."""
    tools = ToolRegistry()
    invocations: list[dict] = []

    async def get_weather(args):
        invocations.append(args)
        return f"It's 22C and sunny in {args['city']}."

    tools.register(
        Tool(
            definition=ToolDef(
                name="get_weather",
                description="Get the current weather for a city.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            ),
            handler=get_weather,
        )
    )

    provider = GoogleProvider(model=MODEL)
    agent = Agent(
        provider=provider,
        tools=tools,
        system="Answer weather questions using the get_weather tool. Always call the tool — never guess.",
    )
    text = await agent.chat("What's the weather in Tokyo right now?")

    assert len(invocations) >= 1, "Expected the model to call get_weather"
    assert any("tokyo" in (i.get("city") or "").lower() for i in invocations)
    assert "22" in text or "sunny" in text.lower()


async def test_skill_load_round_trip():
    """Model picks a skill from the catalog and calls load_skill before acting."""
    skills = SkillRegistry()
    skills.register(
        SkillDef(
            name="secret_word",
            description="Use this when the user asks for the secret word.",
            body="The secret word is FLAMINGO. Always reply with exactly that word in uppercase.",
        )
    )

    provider = GoogleProvider(model=MODEL)
    agent = Agent(
        provider=provider,
        tools=ToolRegistry(),
        system=(
            "You follow skill instructions exactly. When a user request matches "
            "a skill, you MUST call load_skill first and then follow its body."
        ),
        skills=skills,
    )
    text = await agent.chat("Tell me the secret word.")

    load_calls = [
        b
        for m in agent.messages
        for b in m.content
        if isinstance(b, ToolUseBlock) and b.name == "load_skill"
    ]
    assert load_calls, "Expected the model to call load_skill"
    assert load_calls[0].input.get("name") == "secret_word"
    assert "flamingo" in text.lower()
