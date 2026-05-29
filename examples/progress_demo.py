"""Streaming tool progress demo — watch a long-running tool emit progress
events while the agent waits for its result.

Run from the repo root:
    uv run python examples/progress_demo.py

Shows what `ToolProgressEvent`s look like in chat_stream. The same events
render inline in `ambi chat` (italic dim lines inside the panel) and in
the Telegram bot (appended to the progressively-edited reply).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ambi import Agent, Tool, ToolDef, ToolRegistry, load_env
from ambi.providers.google import GoogleProvider
from ambi.types import (
    ChatComplete,
    ToolProgressEvent,
    ToolResultEvent,
    ToolUseEvent,
)


async def slow_processor(args: dict, progress) -> str:
    """A tool that pretends to do work in stages and reports each one."""
    steps = [
        "fetching data from source",
        "parsing 100 items",
        "ranking by relevance",
        "computing summary",
    ]
    for step in steps:
        await progress(step)
        await asyncio.sleep(0.5)
    return f"Processed query '{args.get('query', '?')}' — 100 items summarised."


def build_agent() -> Agent:
    tools = ToolRegistry()
    tools.register(
        Tool(
            definition=ToolDef(
                name="slow_processor",
                description="Pretend to process a query, with mid-flight progress updates.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            ),
            handler=slow_processor,
        )
    )
    return Agent(
        provider=GoogleProvider(model="gemini-2.5-flash"),
        tools=tools,
        system="Use slow_processor when the user asks to process or analyse something.",
    )


async def main() -> None:
    load_env()
    agent = build_agent()
    started = datetime.now()

    async for ev in agent.chat_stream("Process the query 'streaming-tools'"):
        elapsed = (datetime.now() - started).total_seconds()
        prefix = f"[{elapsed:5.2f}s]"
        if isinstance(ev, ToolUseEvent):
            print(f"{prefix} ↳ tool: {ev.name}({ev.input})")
        elif isinstance(ev, ToolProgressEvent):
            print(f"{prefix}    · {ev.message}")
        elif isinstance(ev, ToolResultEvent):
            print(f"{prefix} ✓ result: {ev.content[:80]}")
        elif isinstance(ev, ChatComplete):
            print(f"\n{prefix} Final: {ev.final_text}")


if __name__ == "__main__":
    asyncio.run(main())
