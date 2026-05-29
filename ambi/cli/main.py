"""ambi CLI entry point.

Subcommands:

    ambi init              create ~/.ambi/, seed .env template and example skills
    ambi run               start the Telegram bot + scheduler daemon
    ambi chat              interactive terminal REPL
    ambi version           print version

Long-running daemon (`ambi run`) keeps the Telegram bot and the cron-style
scheduler alive. Once running, the bot is your remote — DM it from your
phone, ask for reminders, run commands, query Hippocamp memory.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import paths


_ENV_TEMPLATE = """# ambi config. Edit this file then run `ambi run`.

# === required ===
GEMINI_API_KEY=

# === Telegram (required for `ambi run`; optional for `ambi chat`) ===
TELEGRAM_BOT_TOKEN=
# Comma-separated Telegram user IDs allowed to talk to the bot.
# Leave empty in dev to allow everyone (NOT recommended in production).
TELEGRAM_ALLOWED_USER_IDS=

# === Hippocamp memory (optional) ===
# pip install ambi-core[hippocamp]
# Then set AMBI_USE_HIPPOCAMP=1. HIPPOCAMP_CMD is only needed if
# hippocamp-mcp isn't on your PATH.
AMBI_USE_HIPPOCAMP=
HIPPOCAMP_CMD=

# === SenseGate ===
# 0 (default) = skip the LLM verifier on read-only turns — cheaper
# 1           = verify every turn with tool calls — full observability
AMBI_VERIFY_READS=0

# === Model selection ===
# AMBI_MODEL=gemini-2.5-flash

# === run_command allowlist ===
# Comma-separated. Override default if you want a different set.
# AMBI_RUN_COMMAND_ALLOW=ls,cat,grep,git
"""


_EXAMPLE_SKILL_TIME = """---
name: time
description: When the user asks about the current time, date, or what day it is.
---

To answer time/date questions, call `get_current_time(timezone)`. Default to
`UTC` if the user doesn't specify. Format the response as a single short
sentence — no markdown, no lists.
"""


_EXAMPLE_SKILL_SHELL = """---
name: shell
description: When the user asks about files, directories, or git state in the current project.
---

To answer questions about the local filesystem or git, call `run_command`
with `argv` as a list — never a single shell string. The tool description
lists which commands are currently allowed; if a command isn't allowed,
say so honestly rather than guessing.

Examples:
- list files:     `run_command({"argv": ["ls", "-la"]})`
- inspect repo:   `run_command({"argv": ["git", "status"]})`
- recent commits: `run_command({"argv": ["git", "log", "--oneline", "-n", "10"]})`
- show a file:    `run_command({"argv": ["cat", "README.md"]})`
- search:         `run_command({"argv": ["grep", "-r", "pattern", "."]})`
"""


def _get_version() -> str:
    try:
        return version("ambi-core")
    except PackageNotFoundError:
        return "0.0.0-dev"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    paths.ensure_tree()
    home = paths.ambi_home()
    env_path = paths.env_file()
    made: list[str] = []

    if not env_path.exists():
        env_path.write_text(_ENV_TEMPLATE)
        made.append(str(env_path))

    skills_dir = paths.skills_dir()
    for name, body in [
        ("time.md", _EXAMPLE_SKILL_TIME),
        ("shell.md", _EXAMPLE_SKILL_SHELL),
    ]:
        skill_path = skills_dir / name
        if not skill_path.exists():
            skill_path.write_text(body)
            made.append(str(skill_path))

    print(f"ambi home: {home}")
    if made:
        print("Created:")
        for p in made:
            print(f"  {p}")
    else:
        print("Already initialized — nothing to do.")
    print()
    print("Next steps:")
    print(f"  1. Edit {env_path} and add GEMINI_API_KEY (and TELEGRAM_BOT_TOKEN if you want the bot)")
    print("  2. Run `ambi chat` for a local REPL, or `ambi run` to start the Telegram daemon")
    return 0


async def _run_chat() -> int:
    from ..env import load_env
    from ..types import Message, TextBlock
    from .build import build_agent

    load_env(paths.env_file())
    paths.ensure_tree()
    agent = build_agent(extra_tools=[], with_hippocamp=False, task_store=None)
    await agent.load()

    print(
        f"ambi REPL — {len(agent.messages)} message(s) loaded from {paths.session_db()}.\n"
        "Type 'exit' to quit, 'history' to dump messages.\n"
    )
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            return 0
        if user_input.lower() == "history":
            for i, m in enumerate(agent.messages):
                print(f"[{i}] {m.role}: {m.content}")
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


async def _run_daemon() -> int:
    from ..env import load_env, require_env
    from ..integrations.hippocamp import hippocamp_server, load_hippocamp_tools
    from ..scheduler import ScheduledTask, Scheduler, TaskStore
    from ..tool import Tool
    from ..transports.telegram import TelegramTransport, split_message
    from .build import build_agent

    load_env(paths.env_file())
    paths.ensure_tree()

    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print(
            f"error: TELEGRAM_BOT_TOKEN not set. Edit {paths.env_file()} or "
            "run `ambi chat` for a local-only session.",
            file=sys.stderr,
        )
        return 2

    task_store = TaskStore(paths.tasks_db())

    async def _go(extra_tools: list[Tool], with_hippocamp: bool) -> None:
        agent = build_agent(extra_tools, with_hippocamp, task_store)
        await agent.load()

        allowed = _parse_allowed_user_ids()
        delivery_chat_id = next(iter(allowed)) if allowed else None

        transport = TelegramTransport(
            agent=agent,
            bot_token=require_env("TELEGRAM_BOT_TOKEN"),
            allowed_user_ids=allowed,
            task_store=task_store,
        )

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
                    print(f"scheduler deliver failed: {e}", file=sys.stderr)

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
        auth_note = (
            f"{len(allowed)} allowed user(s)" if allowed else "DEV MODE: allow all"
        )
        pending = await task_store.list()
        print(
            f"ambi daemon{extras} running. {auth_note}. "
            f"{len(agent.messages)} messages loaded, "
            f"{len(pending)} scheduled task(s) pending. "
            "Ctrl-C to stop."
        )
        try:
            await stop_event.wait()
        finally:
            await scheduler.stop()
            await transport.stop()

    if os.getenv("AMBI_USE_HIPPOCAMP") == "1":
        cmd_raw = os.getenv("HIPPOCAMP_CMD", "hippocamp-mcp")
        parts = cmd_raw.split()
        async with hippocamp_server(
            command=parts[0], args=parts[1:], errlog=paths.hippocamp_log(),
        ) as hippo:
            tools = await load_hippocamp_tools(hippo)
            await _go(tools, with_hippocamp=True)
    else:
        await _go(extra_tools=[], with_hippocamp=False)

    return 0


def _parse_allowed_user_ids() -> set[int] | None:
    raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    if not raw:
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def cmd_run(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_daemon())
    except KeyboardInterrupt:
        return 0


def cmd_chat(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_run_chat())
    except KeyboardInterrupt:
        return 0


def cmd_version(args: argparse.Namespace) -> int:
    print(f"ambi {_get_version()}")
    return 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ambi",
        description=(
            "Personal AI agent harness. Run `ambi init` to set up, then "
            "`ambi run` to start the daemon (Telegram bot + scheduler) or "
            "`ambi chat` for a local REPL."
        ),
    )
    p.add_argument(
        "--version", action="store_true", help="print version and exit",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("init", help="create ~/.ambi/ with .env and example skills").set_defaults(func=cmd_init)
    sub.add_parser("run", help="start the Telegram bot + scheduler daemon").set_defaults(func=cmd_run)
    sub.add_parser("chat", help="local REPL against the same session").set_defaults(func=cmd_chat)
    sub.add_parser("version", help="print version").set_defaults(func=cmd_version)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        return cmd_version(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
