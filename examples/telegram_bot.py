"""Run ambi-core as a Telegram bot.

Required env (in `.env` or environment):
    GEMINI_API_KEY
    TELEGRAM_BOT_TOKEN

Optional:
    TELEGRAM_ALLOWED_USER_IDS  — comma-separated Telegram numeric user IDs;
                                  empty/unset = dev mode (allow everyone)
    AMBI_USE_HIPPOCAMP=1       — attach long-term memory
    HIPPOCAMP_CMD              — Hippocamp MCP launch command

Shares the SQLite session db with the REPL — what you say on Telegram is
recallable from the REPL and vice-versa.
"""

from __future__ import annotations

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ambi import (
    Agent,
    CommandPolicy,
    LLMClaimVerifier,
    ScheduledTask,
    Scheduler,
    SenseGate,
    SkillRegistry,
    SqliteStore,
    TaskStore,
    Tool,
    ToolDef,
    ToolRegistry,
    load_env,
    make_run_command_tool,
    make_scheduler_tools,
    require_env,
)
from ambi.integrations.hippocamp import hippocamp_server, load_hippocamp_tools
from ambi.providers.google import GoogleProvider
from google.genai import types as gt
from ambi.transports.telegram import TelegramTransport, split_message

REPO_ROOT = Path(__file__).parent.parent
SESSION_DB = REPO_ROOT / "data" / "session.db"
TASKS_DB = REPO_ROOT / "data" / "tasks.db"
HIPPOCAMP_LOG = REPO_ROOT / "data" / "hippocamp.log"

COMMAND_ALLOWLIST = {
    "ls", "pwd", "cat", "head", "tail", "wc", "find",
    "git", "date", "echo", "grep",
}

SYSTEM_BASE = (
    "You are a concise assistant talking to the user over Telegram. Keep "
    "replies short — one or two short paragraphs unless the user explicitly "
    "asks for detail. Use plain text; minimal markdown.\n\n"
    "You can self-schedule via the `schedule` tool. Use it when the user "
    "asks for a reminder, a recurring routine, or a future check-in. Pass "
    "run_at as an ISO 8601 UTC timestamp (call get_current_time first if "
    "unsure of 'now'). Use `cron` for recurring tasks. The scheduled prompt "
    "you set will run as your future self with the same tools — write it "
    "as a directive (e.g. 'Summarize my GitHub activity from the last 24h "
    "and report key items.')."
)
SYSTEM_HIPPOCAMP_ADDON = (
    "\n\nYou have access to Hippocamp memory tools (recall_memory, "
    "update_memory, etc.). Use `recall_memory` proactively when the user "
    "references past context. Use `update_memory` to save stable facts, "
    "preferences, or decisions."
)


async def _get_current_time(args: dict) -> str:
    tz_name = (args.get("timezone") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Error: unknown timezone '{tz_name}'"
    return datetime.now(tz).strftime("%A %Y-%m-%d %H:%M:%S %Z")


def _build_agent(
    extra_tools: list[Tool], with_hippocamp: bool, task_store: TaskStore,
) -> Agent:
    tools = ToolRegistry()
    for t in make_scheduler_tools(task_store):
        tools.register(t)
    tools.register(Tool(
        definition=ToolDef(
            name="get_current_time",
            description="Get the current date and time in a given IANA timezone.",
            input_schema={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone, e.g. 'UTC'"},
                },
                "required": [],
            },
        ),
        handler=_get_current_time,
    ))
    tools.register(make_run_command_tool(CommandPolicy(
        allowed=COMMAND_ALLOWLIST,
        cwd_root=REPO_ROOT,
        default_timeout=15.0,
        max_output_bytes=20_000,
    )))
    for t in extra_tools:
        tools.register(t)

    skills_dir = REPO_ROOT / "examples" / "skills"
    skills = SkillRegistry.from_dir(skills_dir)

    provider = GoogleProvider(model="gemini-2.5-flash")
    # Verifier runs without thinking — match/mismatch is pattern-matching,
    # not reasoning. Thinking otherwise burns most of the token budget on
    # invisible reasoning. Off → ~30x output cost reduction per call.
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


def _parse_allowed_user_ids() -> set[int] | None:
    raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    if not raw:
        return None  # allow all (dev mode)
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


async def _run(extra_tools: list[Tool], with_hippocamp: bool) -> None:
    task_store = TaskStore(TASKS_DB)
    agent = _build_agent(extra_tools, with_hippocamp, task_store)
    await agent.load()

    transport = TelegramTransport(
        agent=agent,
        bot_token=require_env("TELEGRAM_BOT_TOKEN"),
        allowed_user_ids=_parse_allowed_user_ids(),
        task_store=task_store,
    )

    allowed = transport.allowed_user_ids
    delivery_chat_id = next(iter(allowed)) if allowed else None

    async def _deliver_scheduled(task: ScheduledTask, reply: str) -> None:
        if delivery_chat_id is None or transport._app is None:
            return
        header = f"⏰ scheduled run (task {task.id})\n\n"
        for chunk in split_message(header + reply):
            try:
                await transport._app.bot.send_message(
                    chat_id=delivery_chat_id, text=chunk,
                )
            except Exception as e:
                print(f"scheduler deliver failed: {e}")

    scheduler = Scheduler(
        store=task_store,
        agent=agent,
        on_result=_deliver_scheduled,
        check_interval=15.0,
    )

    stop_event = asyncio.Event()

    def _handle_signal():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    await transport.start()
    await scheduler.start()
    extras = " + hippocamp" if with_hippocamp else ""
    auth_note = f"{len(allowed)} allowed user(s)" if allowed else "DEV MODE: allow all"
    pending = await task_store.list()
    print(
        f"ambi telegram bot{extras} + scheduler running. {auth_note}. "
        f"{len(agent.messages)} messages loaded, {len(pending)} scheduled task(s) pending. "
        "Ctrl-C to stop."
    )
    try:
        await stop_event.wait()
    finally:
        await scheduler.stop()
        await transport.stop()


async def main() -> None:
    load_env()
    if os.getenv("AMBI_USE_HIPPOCAMP") == "1":
        # Default: hippocamp-mcp entry point on PATH (from
        # `pip install ambi-core[hippocamp]`). Override HIPPOCAMP_CMD only
        # if you're using a non-PATH venv.
        cmd_raw = os.getenv("HIPPOCAMP_CMD", "hippocamp-mcp")
        parts = cmd_raw.split()
        async with hippocamp_server(
            command=parts[0], args=parts[1:], errlog=HIPPOCAMP_LOG,
        ) as hippo:
            tools = await load_hippocamp_tools(hippo)
            await _run(tools, with_hippocamp=True)
    else:
        await _run(extra_tools=[], with_hippocamp=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
