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


def _load_runtime_env() -> None:
    """Load env from ~/.ambi/.env then a project-local .env (overrides).

    Running `ambi chat` / `ambi run` from a project directory with a
    local .env (the dev case) lets you experiment without touching your
    daily ~/.ambi/.env. Outside a project, only ~/.ambi/.env is loaded.
    """
    from ..env import load_env
    load_env(paths.env_file())
    local = Path.cwd() / ".env"
    if local.exists() and local.resolve() != paths.env_file().resolve():
        load_env(local, override=True)


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


_EXAMPLE_SKILL_OBSIDIAN = """---
name: obsidian
description: Read, search, write, or delete notes in the user's Obsidian vault (PARA-organized).
---

The Obsidian vault is the user's second brain — a directory tree of
markdown files organised by **PARA**:

- **Projects/** — active commitments with a deadline or clear outcome
- **Areas/** — ongoing responsibilities (Work, Family, Health, Finances…)
- **Resources/** — reference material, learning, things to revisit
- **Archive/** — completed projects, retired areas, anything inactive
- **Inbox/** — capture buffer; everything new lands here and gets sorted later

Ground every claim about vault content in a tool call. Never invent note
contents or paths.

## Default save → Inbox

When the user asks you to save, jot, note, or capture something, write to
**Inbox** unless they explicitly name a different folder. Don't try to be
clever about filing on first capture — the user will sort it themselves
during their PARA review. Calling `obsidian_save` without a `folder` arg
lands in Inbox automatically; the tool description confirms the default.

## When to specify a different folder

Only file directly into a PARA bucket when the user's intent is clear:
- "Add this to my Postgres migration project" → `folder: "Projects/Postgres migration"`
- "Save this under my Health area" → `folder: "Areas/Health"`
- "Archive this old project note" → move to `Archive/<original-folder>`

For anything else, prefer Inbox.

## Typical flow

- **Search first**: `obsidian_search({"query": "..."})` — full-text across
  filenames and bodies. Falls back to `obsidian_list` for browsing a folder.
- **Read**: `obsidian_read({"path": "Areas/Work/MOC.md"})` — full content,
  including frontmatter.
- **Save (capture)**: `obsidian_save({"title": "...", "content": "...", "tags": "..."})`
  → lands in Inbox.
- **Save (filed)**: same call with `"folder": "Areas/Health"` when intent is clear.
- **Delete**: destructive — confirm with the user first if the intent is ambiguous.

Always pass the full markdown body in `content` — don't truncate or
summarise the user's input.
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
    from .build import SYSTEM_BASE

    paths.ensure_tree()
    home = paths.ambi_home()
    env_path = paths.env_file()
    made: list[str] = []

    if not env_path.exists():
        env_path.write_text(_ENV_TEMPLATE)
        made.append(str(env_path))

    system_path = paths.system_md()
    if not system_path.exists():
        system_path.write_text(SYSTEM_BASE + "\n")
        made.append(str(system_path))

    skills_dir = paths.skills_dir()
    for name, body in [
        ("time.md", _EXAMPLE_SKILL_TIME),
        ("shell.md", _EXAMPLE_SKILL_SHELL),
        ("obsidian.md", _EXAMPLE_SKILL_OBSIDIAN),
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
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from rich.live import Live

    from ..integrations.hippocamp import hippocamp_server, load_hippocamp_tools
    from ..types import (
        ChatComplete,
        SenseGateFlagEvent,
        TextBlock,
        TextDelta,
        ToolResultBlock,
        ToolResultEvent,
        ToolUseBlock,
        ToolUseEvent,
    )
    from .build import build_agent

    from ..scheduler import TaskStore

    _load_runtime_env()
    paths.ensure_tree()

    console = Console()
    # Share the same tasks.db as `ambi run` so schedule() calls from the
    # REPL persist; the daemon fires them when it's up.
    task_store = TaskStore(paths.tasks_db())
    with_hippocamp = os.getenv("AMBI_USE_HIPPOCAMP") == "1"

    async def _build_with_optional_hippocamp():
        if not with_hippocamp:
            agent = build_agent(
                extra_tools=[], with_hippocamp=False, task_store=task_store,
            )
            await agent.load()
            return agent, None
        cmd_raw = os.getenv("HIPPOCAMP_CMD", "hippocamp-mcp")
        parts = cmd_raw.split()
        hippo_cm = hippocamp_server(
            command=parts[0], args=parts[1:], errlog=paths.hippocamp_log(),
        )
        hippo = await hippo_cm.__aenter__()
        try:
            hippo_tools = await load_hippocamp_tools(hippo)
            agent = build_agent(
                extra_tools=hippo_tools, with_hippocamp=True, task_store=task_store,
            )
            await agent.load()
            return agent, hippo_cm
        except BaseException:
            await hippo_cm.__aexit__(None, None, None)
            raise

    agent, hippo_cm = await _build_with_optional_hippocamp()

    async def _close_hippo():
        if hippo_cm is not None:
            await hippo_cm.__aexit__(None, None, None)

    history_file = paths.ambi_home() / ".chat_history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        # Multiline pastes arrive atomically thanks to bracketed paste mode
        # (terminal-emulator feature, detected by prompt_toolkit by default).
    )

    banner = Text()
    banner.append("ambi ", style="bold magenta")
    banner.append(f"v{_get_version()}\n", style="dim")
    banner.append(f"Session: {len(agent.messages)} messages · ", style="dim")
    banner.append(str(paths.session_db()), style="dim cyan")
    banner.append("\nCommands: ", style="dim")
    banner.append("exit", style="dim bold")
    banner.append(" · ", style="dim")
    banner.append("history", style="dim bold")
    banner.append(" · ", style="dim")
    banner.append("audit", style="dim bold")
    console.print(Panel(banner, border_style="cyan", padding=(0, 1)))
    console.print()

    try:
        while True:
            try:
                user_input = (
                    await session.prompt_async(
                        HTML("<ansigreen><b>❯ </b></ansigreen>")
                    )
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return 0
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                return 0
            if user_input.lower() == "history":
                _render_history(console, agent.messages)
                continue
            if user_input.lower() == "audit":
                _render_audit(console, agent)
                continue

            snapshot = len(agent.messages)
            try:
                await _stream_turn(console, agent, user_input)
            except (KeyboardInterrupt, asyncio.CancelledError):
                del agent.messages[snapshot:]
                console.print("[dim italic](cancelled)[/dim italic]")
                continue
            except Exception as e:
                del agent.messages[snapshot:]
                console.print(f"[red]error: {type(e).__name__}: {e}[/red]")
                continue
            console.print()
    finally:
        await _close_hippo()


async def _stream_turn(console, agent, user_input: str) -> None:
    """Drive agent.chat_stream() through a rich Live panel."""
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text

    from ..types import (
        ChatComplete,
        SenseGateFlagEvent,
        TextDelta,
        ToolResultEvent,
        ToolUseEvent,
    )

    buffer = ""
    panel_title = "[bold magenta]ambi[/bold magenta]"

    def panel() -> Panel:
        body = Markdown(buffer) if buffer else Spinner("dots", text="thinking…")
        return Panel(
            body,
            title=panel_title,
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
        )

    with Live(panel(), console=console, refresh_per_second=12, transient=False) as live:
        async for ev in agent.chat_stream(user_input):
            if isinstance(ev, TextDelta):
                buffer += ev.text
                live.update(panel())
            elif isinstance(ev, ToolUseEvent):
                # Tool calls between text chunks — note them inside the panel as
                # dim cyan lines so the user sees activity without losing the
                # accumulated text.
                buffer += f"\n\n_↳ {ev.name}({_short_args(ev.input, 60)})_\n"
                live.update(panel())
            elif isinstance(ev, ToolResultEvent):
                if ev.is_error:
                    buffer += f"_  ✗ {ev.name} error_\n"
                    live.update(panel())
            elif isinstance(ev, SenseGateFlagEvent):
                buffer += (
                    f"\n_⚠ SenseGate flagged the prior reply — "
                    f"{ev.reason[:120]}. Restating._\n\n"
                )
                live.update(panel())
            elif isinstance(ev, ChatComplete):
                # Stream finished — the buffer already holds the final visible
                # text. Live will exit with the current panel snapshot.
                pass


def _render_tool_trace(console, new_messages) -> None:
    """Show a one-line summary of any tools called in this turn."""
    from ..types import ToolResultBlock, ToolUseBlock

    lines = []
    for m in new_messages:
        if m.role != "assistant":
            continue
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                args_preview = _short_args(b.input)
                lines.append(f"  ↳ {b.name}({args_preview})")
    if not lines:
        return
    from rich.text import Text
    trace = Text("\n".join(lines), style="dim cyan")
    console.print(trace)


def _short_args(d: dict, limit: int = 60) -> str:
    if not d:
        return ""
    import json
    raw = json.dumps(d, ensure_ascii=False, default=str)
    if len(raw) > limit:
        raw = raw[: limit - 1] + "…"
    return raw


def _render_history(console, messages) -> None:
    from rich.table import Table

    if not messages:
        console.print("[dim](no messages yet)[/dim]")
        return
    table = Table(
        title=f"Session history · {len(messages)} message(s)",
        show_lines=False,
        header_style="bold cyan",
        title_style="bold",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="cyan", width=10)
    table.add_column("Content")
    for i, m in enumerate(messages):
        table.add_row(str(i), m.role, _summarize_blocks(m.content))
    console.print(table)


def _summarize_blocks(content) -> str:
    from ..types import TextBlock, ToolResultBlock, ToolUseBlock

    parts: list[str] = []
    for b in content:
        if isinstance(b, TextBlock):
            text = b.text.replace("\n", " ")
            parts.append(text[:140] + ("…" if len(text) > 140 else ""))
        elif isinstance(b, ToolUseBlock):
            parts.append(f"[cyan]→ {b.name}({_short_args(b.input, 40)})[/cyan]")
        elif isinstance(b, ToolResultBlock):
            tag = "[red]ERROR[/red]" if b.is_error else "[green]ok[/green]"
            c = b.content if isinstance(b.content, str) else str(b.content)
            c = c.replace("\n", " ")
            parts.append(f"{tag} {c[:120]}{'…' if len(c) > 120 else ''}")
    return " · ".join(parts) if parts else ""


def _render_audit(console, agent) -> None:
    from rich.table import Table

    gate = getattr(agent, "sensegate", None)
    if gate is None or not gate.audit_log:
        console.print("[dim](no audit entries)[/dim]")
        return
    table = Table(
        title=f"SenseGate audit · {len(gate.audit_log)} flag(s)",
        header_style="bold cyan",
        title_style="bold yellow",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Time", style="dim", width=10)
    table.add_column("Kind", width=6)
    table.add_column("Reason")
    table.add_column("Claim")
    for i, e in enumerate(gate.audit_log):
        kind = "[red]WRITE[/red]" if e.had_write else "[yellow]READ[/yellow]"
        table.add_row(
            str(i),
            e.timestamp.strftime("%H:%M:%S"),
            kind,
            (e.reason[:80] + "…") if len(e.reason) > 80 else e.reason,
            e.final_text_excerpt[:80] + ("…" if len(e.final_text_excerpt) > 80 else ""),
        )
    console.print(table)


async def _run_daemon() -> int:
    from ..env import require_env
    from ..integrations.hippocamp import hippocamp_server, load_hippocamp_tools
    from ..scheduler import ScheduledTask, Scheduler, TaskStore
    from ..tool import Tool
    from ..transports.telegram import TelegramTransport, split_message
    from .build import build_agent

    _load_runtime_env()
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
