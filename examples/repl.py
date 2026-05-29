"""Tiny REPL — chat with a Gemini-backed agent.

Run from the repo root:
    uv run python examples/repl.py

Requires GEMINI_API_KEY in `.env` (or the environment).

Optional: set AMBI_USE_HIPPOCAMP=1 to attach Hippocamp long-term memory
(requires the `hippocamp` MCP server on PATH).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ambi import (
    Agent,
    CommandPolicy,
    LLMClaimVerifier,
    SenseGate,
    SkillRegistry,
    SqliteStore,
    Tool,
    ToolDef,
    ToolRegistry,
    load_env,
    make_run_command_tool,
)
from ambi.integrations.hippocamp import hippocamp_server, load_hippocamp_tools
from ambi.providers.google import GoogleProvider
from google.genai import types as gt

REPO_ROOT = Path(__file__).parent.parent
SESSION_DB = REPO_ROOT / "data" / "session.db"
HIPPOCAMP_LOG = REPO_ROOT / "data" / "hippocamp.log"

COMMAND_ALLOWLIST = {
    "ls", "pwd", "cat", "head", "tail", "wc", "find",
    "git", "date", "echo", "grep",
}

SYSTEM_BASE = (
    "You are a concise assistant. Use tools and skills when relevant; otherwise "
    "answer directly in one or two sentences."
)

SYSTEM_HIPPOCAMP_ADDON = (
    "\n\nYou have access to Hippocamp memory tools (recall_memory, update_memory, etc.). "
    "Use `recall_memory` proactively when the user references past context. "
    "Use `update_memory` to save stable facts, preferences, or decisions — not transient state."
)


async def _get_current_time(args: dict) -> str:
    tz_name = (args.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Error: unknown timezone '{tz_name}'"
    now = datetime.now(tz)
    return now.strftime("%A %Y-%m-%d %H:%M:%S %Z")


def _build_agent(extra_tools: list[Tool], with_hippocamp: bool) -> Agent:
    tools = ToolRegistry()
    tools.register(
        Tool(
            definition=ToolDef(
                name="get_current_time",
                description="Get the current date and time in a given IANA timezone.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone, e.g. 'UTC', 'Europe/London'",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_get_current_time,
        )
    )

    tools.register(
        make_run_command_tool(
            CommandPolicy(
                allowed=COMMAND_ALLOWLIST,
                cwd_root=REPO_ROOT,
                default_timeout=15.0,
                max_output_bytes=20_000,
            )
        )
    )

    for t in extra_tools:
        tools.register(t)

    skills_dir = Path(__file__).parent / "skills"
    skills = SkillRegistry.from_dir(skills_dir)

    provider = GoogleProvider(model="gemini-2.5-flash")
    # Verifier deliberately runs with thinking disabled — the JSON
    # match/mismatch judgement is pattern-matching, not reasoning, and
    # thinking burns most of the 2048-token budget invisibly. Off → ~30x
    # output cost reduction per verification.
    # AMBI_VERIFY_READS=1 enables read-side audit logging at the cost of
    # one LLM call per read-only turn. Default skips for cost reasons.
    verify_reads = os.getenv("AMBI_VERIFY_READS", "0") == "1"
    gate = SenseGate(
        verifier=LLMClaimVerifier(
            provider=provider,
            max_tokens=256,
            thinking_config=gt.ThinkingConfig(thinking_budget=0),
        ),
        max_retries=2,
        verify_reads=verify_reads,
    )

    system = SYSTEM_BASE + (SYSTEM_HIPPOCAMP_ADDON if with_hippocamp else "")

    return Agent(
        provider=provider,
        tools=tools,
        system=system,
        skills=skills,
        sensegate=gate,
        store=SqliteStore(SESSION_DB),
    )


async def _run_loop(extra_tools: list[Tool], with_hippocamp: bool) -> None:
    agent = _build_agent(extra_tools, with_hippocamp)
    await agent.load()

    extras = " + hippocamp" if with_hippocamp else ""
    print(
        f"ambi REPL{extras} — 'exit' to quit, 'history' to dump messages, "
        "'audit' to view SenseGate flags.\n"
        f"({len(agent.messages)} message{'s' if len(agent.messages) != 1 else ''} loaded from {SESSION_DB})\n"
    )
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            return
        if user_input.lower() == "history":
            for i, m in enumerate(agent.messages):
                print(f"[{i}] {m.role}: {m.content}")
            continue
        if user_input.lower() == "audit":
            if agent.sensegate is None or not agent.sensegate.audit_log:
                print("(no audit entries)")
            else:
                for i, e in enumerate(agent.sensegate.audit_log):
                    flag = "WRITE" if e.had_write else "READ"
                    print(f"[{i}] {e.timestamp:%H:%M:%S} {flag}  {e.reason}")
                    print(f"    claim: {e.final_text_excerpt[:120]}")
            continue

        snapshot = len(agent.messages)
        try:
            reply = await agent.chat(user_input)
        except (KeyboardInterrupt, asyncio.CancelledError):
            del agent.messages[snapshot:]
            print("\n(cancelled)")
            continue
        except Exception as e:
            del agent.messages[snapshot:]
            print(f"error: {e}")
            continue
        print(f"ambi> {reply}\n")


async def main() -> None:
    load_env()
    if os.getenv("AMBI_USE_HIPPOCAMP") == "1":
        # Default: hippocamp-mcp entry point on PATH (installed via
        # `pip install ambi-core[hippocamp]`). Override HIPPOCAMP_CMD only
        # if you're running from a venv that's not on PATH.
        cmd_raw = os.getenv("HIPPOCAMP_CMD", "hippocamp-mcp")
        parts = cmd_raw.split()
        async with hippocamp_server(
            command=parts[0], args=parts[1:], errlog=HIPPOCAMP_LOG,
        ) as hippo:
            tools = await load_hippocamp_tools(hippo)
            await _run_loop(tools, with_hippocamp=True)
    else:
        await _run_loop(extra_tools=[], with_hippocamp=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
