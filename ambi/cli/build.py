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

    vault = os.getenv("OBSIDIAN_VAULT")
    if vault:
        from ..integrations.obsidian import VaultError, make_obsidian_tools
        default_folder = os.getenv("OBSIDIAN_DEFAULT_FOLDER", "Inbox")
        try:
            for t in make_obsidian_tools(vault, default_folder=default_folder):
                tools.register(t)
        except VaultError as e:
            # Vault misconfigured — log to stderr but don't crash the agent.
            import sys
            print(f"warning: obsidian tools not registered ({e})", file=sys.stderr)

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
