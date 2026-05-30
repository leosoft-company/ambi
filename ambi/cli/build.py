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
from importlib.resources import files
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

def _bundled(name: str) -> str:
    """Read a bundled markdown file shipped alongside this module."""
    return (files("ambi.cli") / name).read_text(encoding="utf-8")


# Personality lives in ``ambi/cli/system.md`` — edit that file (not this
# constant) to change the bundled default. Hippocamp addon lives in
# ``ambi/cli/system_hippocamp.md`` and is appended at runtime when the
# memory tools are wired in.
SYSTEM_BASE = _bundled("system.md").rstrip()
SYSTEM_HIPPOCAMP_ADDON = "\n\n" + _bundled("system_hippocamp.md").rstrip()


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


def _system_time_suffix() -> str:
    """Fresh per-turn timestamp appended to the system prompt so the model
    always knows the current local time (e.g. for time-of-day greetings).
    """
    now = datetime.now().astimezone()
    return f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M %Z')}"


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

    # Each bundled skill that ships a `tools.py` gets a chance to wire its
    # tools into the registry. Skills self-decide based on env vars (e.g.
    # OBSIDIAN_VAULT) — no per-skill plumbing needed here.
    from ..skills import register_bundled_skill_tools
    register_bundled_skill_tools(tools)

    for t in extra_tools:
        tools.register(t)

    # Load bundled package skills first, then layer user skills on top.
    # Same-name user skill wins (override the bundled default).
    skills = SkillRegistry.from_dirs(
        SkillRegistry.bundled_dir(),
        paths.skills_dir(),
    )

    raw_provider = GoogleProvider(model=os.getenv("AMBI_MODEL", DEFAULT_MODEL))
    # Track tokens + cost across chat, sensegate, and compaction calls.
    from ..usage import TrackingProvider, UsageStore
    usage_store = UsageStore(paths.usage_db())
    provider = TrackingProvider(inner=raw_provider, store=usage_store)

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

    compaction_threshold = int(os.getenv("AMBI_COMPACTION_THRESHOLD", "15"))
    context_window_turns = int(os.getenv("AMBI_CONTEXT_WINDOW_TURNS", "20"))

    # Warden: pre-execution policy enforcement. Defaults are conservative:
    #   - ArgvValidator blocks the truly destructive run_command argv shapes.
    #   - CostCeiling caps daily spend (uses the same UsageStore we already
    #     wired for token tracking).
    # No QuietHoursPolicy by default — opt in via env if you want it.
    from ..warden import ArgvValidatorPolicy, CostCeilingPolicy, Warden
    warden = Warden(policies=[
        ArgvValidatorPolicy(forbid=[
            "git push --force",
            "git push -f",
            "git reset --hard origin",
            "rm -rf /",
            "rm -rf ~",
            "rm -rf $HOME",
        ]),
        CostCeilingPolicy(
            usage_store=usage_store,
            daily_usd=float(os.getenv("AMBI_DAILY_COST_USD", "1.00")),
        ),
    ])

    # Generous max_output_tokens + explicit thinking budget so Gemini's
    # invisible reasoning doesn't eat the whole response budget. ambi-lite
    # showed that 4096-thinking + 65536-output is the reliability win;
    # 16384 output is enough for most replies while keeping cost bounded.
    main_max_tokens = int(os.getenv("AMBI_MAX_TOKENS", "16384"))
    main_thinking_budget = int(os.getenv("AMBI_THINKING_BUDGET", "4096"))

    return Agent(
        provider=provider,
        tools=tools,
        system=system,
        skills=skills,
        sensegate=gate,
        store=SqliteStore(paths.session_db()),
        compaction_threshold=compaction_threshold,
        context_window_turns=context_window_turns,
        warden=warden,
        max_tokens=main_max_tokens,
        provider_kwargs={
            "thinking_config": gt.ThinkingConfig(
                thinking_budget=main_thinking_budget,
            ),
        },
        system_suffix_fn=_system_time_suffix,
    )
