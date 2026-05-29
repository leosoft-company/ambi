"""Shared agent + scheduler factory used by `ambi run` and `ambi chat`.

Reads config from environment (which `load_env(env_file())` populates from
``~/.ambi/.env``). Produces a fully-wired `Agent` with default tools and
optional Hippocamp memory.

The example scripts under ``examples/`` show the raw library API; this
module is the opinionated one for the installed CLI.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from google.genai import types as gt

from ..loop import Agent
from ..providers.google import GoogleProvider
from ..run_command import CommandPolicy, make_run_command_tool
from ..scheduler import TaskStore
from ..sensegate import LLMClaimVerifier, SenseGate
from ..skills import SkillRegistry, make_load_skill_tool
from ..store import SqliteStore
from ..tool import Tool, ToolRegistry
from ..types import ToolDef
from . import paths

DEFAULT_MODEL = "gemini-2.5-flash"

# Read-mostly default allowlist for run_command. Users tighten/extend via
# the AMBI_RUN_COMMAND_ALLOW env var (comma-separated).
DEFAULT_COMMAND_ALLOWLIST = {
    "ls", "pwd", "cat", "head", "tail", "wc", "find",
    "git", "date", "echo", "grep",
}

SYSTEM_BASE = """\
You are ambi — a personal AI assistant. Your baseline is competent and \
brief, like a sharp colleague who knows the codebase and assumes the user \
is busy. Speak in clean declarative sentences. No padding intros, no \
closing flourishes, no "happy to help" / "let me know if" / "sure!".

When you act, quote receipts — IDs, file paths, exact values — from the \
tool results. Never paraphrase away a failure: if a tool returned an \
error, say so plainly. "Couldn't reach the API — connection refused." not \
"Oh no, it seems there was an issue!".

You're observant. Once in a while — when there's real signal — surface a \
pattern or assumption worth flagging: a recurring request that should be \
scheduled, a contradiction with something the user said earlier, an \
anomalous tool result. The bar is "would a thoughtful colleague mention \
this?" Most turns don't need it.

Don't apologise for the model's limits. Don't ask permission for routine \
work. Don't triple-check before acting on a clear request.

Scheduling: you can self-schedule via the `schedule` tool. Use it for \
reminders, recurring routines, and future check-ins. Pass `run_at` as an \
ISO 8601 UTC timestamp (call `get_current_time` first if you don't know \
"now"). Use `cron` for recurring tasks. The scheduled prompt you set will \
run as your future self with the same tools — write it as a directive.\
"""


SYSTEM_HIPPOCAMP_ADDON = """\

Memory: you have `recall_memory` and `update_memory` (Hippocamp). Recall \
proactively when the user references past context — don't make them \
remind you. Save stable facts, preferences, decisions, and notable events \
when they happen; don't save transient state. When you reference \
something you remembered, name the source ("From memory: …") — don't \
pretend to know things you didn't actually recall.\
"""


def load_system_prompt(with_hippocamp: bool) -> str:
    """Return the system prompt. Reads ~/.ambi/system.md if present, else
    falls back to the bundled default. Hippocamp addon is appended at the
    end when enabled.
    """
    override = paths.system_md()
    base = override.read_text() if override.exists() else SYSTEM_BASE
    if with_hippocamp:
        return base + SYSTEM_HIPPOCAMP_ADDON
    return base


async def _get_current_time(args: dict) -> str:
    tz_name = (args.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Error: unknown timezone '{tz_name}'"
    return datetime.now(tz).strftime("%A %Y-%m-%d %H:%M:%S %Z")


def _time_tool() -> Tool:
    return Tool(
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


def _command_allowlist() -> set[str]:
    raw = os.getenv("AMBI_RUN_COMMAND_ALLOW", "").strip()
    if not raw:
        return set(DEFAULT_COMMAND_ALLOWLIST)
    return {x.strip() for x in raw.split(",") if x.strip()}


def build_agent(
    extra_tools: list[Tool],
    with_hippocamp: bool,
    task_store: TaskStore | None,
) -> Agent:
    """Wire up an Agent with the default tool stack, SenseGate, and store."""
    tools = ToolRegistry()
    tools.register(_time_tool())
    tools.register(make_run_command_tool(CommandPolicy(
        allowed=_command_allowlist(),
        cwd_root=None,  # CLI users get free filesystem access; tighten via env if needed
        default_timeout=15.0,
        max_output_bytes=20_000,
    )))
    if task_store is not None:
        from ..scheduler import make_scheduler_tools
        for t in make_scheduler_tools(task_store):
            tools.register(t)
    for t in extra_tools:
        tools.register(t)

    skills = SkillRegistry.from_dir(paths.skills_dir())

    provider = GoogleProvider(model=os.getenv("AMBI_MODEL", DEFAULT_MODEL))
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

    system = load_system_prompt(with_hippocamp=with_hippocamp)

    return Agent(
        provider=provider,
        tools=tools,
        system=system,
        skills=skills,
        sensegate=gate,
        store=SqliteStore(paths.session_db()),
    )
