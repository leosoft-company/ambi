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

# === Output + thinking budget (per chat turn) ===
# Gemini's thinking tokens count against max_output_tokens. With a small
# budget, thinking can eat the whole budget and the visible reply ends
# up empty. Defaults give thinking + reply both plenty of room.
# AMBI_MAX_TOKENS=16384
# AMBI_THINKING_BUDGET=4096

# === Context window + compaction ===
# How many of the most-recent user-text turns are kept verbatim in the
# LLM-facing slice (with all assistant + tool messages between them).
# Each turn ~3 messages with tool use; cost scales linearly.
# AMBI_CONTEXT_WINDOW_TURNS=20
#
# Once this many user-text turns accumulate beyond the verbatim window,
# they get summarized into a single anchor for long-term recall. 0 = off.
# AMBI_COMPACTION_THRESHOLD=15

# === Observability ===
# Log level for the `ambi` logger (DEBUG/INFO/WARNING/ERROR). Logs go to
# ~/.ambi/logs/ambi.log (rotating) and, for `ambi run`, also to stderr.
# Per-turn telemetry is recorded to ~/.ambi/data/telemetry.db — inspect with
# `ambi logs` (recent turns) and `ambi status` (health metrics).
# AMBI_LOG_LEVEL=INFO

# === run_command allowlist ===
# Comma-separated. Override default if you want a different set.
# AMBI_RUN_COMMAND_ALLOW=ls,cat,grep,git

# === Egress allowlist ===
# Hosts run_command may push/clone to. Defaults to github.com, gitlab.com,
# bitbucket.org; pushes/remotes to any other host are denied. Add your own
# (e.g. self-hosted git) here, comma-separated.
# AMBI_ALLOWED_GIT_HOSTS=git.mycompany.com
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

    # Built-in skills (time, shell, obsidian, …) ship with the ambi-core
    # package under ambi/skills/. We don't copy them here — they're loaded
    # directly. ~/.ambi/skills/ is just for user overrides and additions.

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
    # REPL: log to file only — stderr would fight the rich Live panel.
    from ..observability import setup_logging
    setup_logging(log_dir=paths.logs_dir(), stderr=False)

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

    # Human-in-the-loop confirmer for Warden require_confirmation verdicts
    # (egress: git push, new remotes, …). The Live panel handle is injected
    # per-turn via this holder so the confirmer can pause it to read stdin.
    live_holder: dict[str, object] = {"live": None}
    agent.confirm = _make_confirmer(console, live_holder)

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
    banner.append(" · ", style="dim")
    banner.append("usage", style="dim bold")
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
            if user_input.lower() == "usage":
                await _render_usage_inline(console)
                continue

            snapshot = len(agent.messages)
            try:
                await _stream_turn(console, agent, user_input, live_holder)
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


def _make_confirmer(console, live_holder: dict):
    """Build an interactive y/N confirmer for Warden require_confirmation.

    Pauses the active Live panel (if any) so the prompt reads cleanly from
    stdin, then resumes. Anything other than an explicit yes is a decline —
    the loop fails closed.
    """

    async def confirm(ctx, decision) -> bool:
        live = live_holder.get("live")
        if live is not None:
            live.stop()
        try:
            console.print(
                f"\n[bold yellow]⚠ confirm[/bold yellow] "
                f"[cyan]{ctx.tool_name}[/cyan] — {decision.reason}"
            )
            console.print(f"  [dim]{_short_args(ctx.tool_input, 200)}[/dim]")
            try:
                answer = console.input("  Allow this action? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = ""
        finally:
            if live is not None:
                live.start()
        return answer.strip().lower() in {"y", "yes"}

    return confirm


async def _stream_turn(console, agent, user_input: str, live_holder=None) -> None:
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
        ToolProgressEvent,
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
        if live_holder is not None:
            live_holder["live"] = live
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
            elif isinstance(ev, ToolProgressEvent):
                buffer += f"_   · {ev.message}_\n"
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
    # Daemon: log to both the rotating file and stderr so `ambi run` output
    # shows what the agent is doing (turns, tool calls, warden decisions).
    from ..observability import setup_logging
    setup_logging(log_dir=paths.logs_dir(), stderr=True)

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
        if not allowed:
            print(
                "⚠ TELEGRAM_ALLOWED_USER_IDS is empty — the bot will accept "
                "messages from ANYONE who finds it. Set it to your numeric "
                "Telegram user id before exposing the bot publicly.",
                file=sys.stderr,
            )
        delivery_chat_id = next(iter(allowed)) if allowed else None

        transport = TelegramTransport(
            agent=agent,
            bot_token=require_env("TELEGRAM_BOT_TOKEN"),
            allowed_user_ids=allowed,
            task_store=task_store,
            telemetry_store=agent.telemetry,
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


def cmd_usage(args: argparse.Namespace) -> int:
    """Show token/cost summary: today, this week, all-time."""
    return asyncio.run(_run_usage(args))


async def _run_usage(args: argparse.Namespace) -> int:
    from datetime import datetime, timedelta, timezone

    from rich.console import Console

    from ..usage import UsageStore

    store = UsageStore(paths.usage_db())
    console = Console()
    now = datetime.now(timezone.utc)

    windows = [
        ("Today", now.replace(hour=0, minute=0, second=0, microsecond=0)),
        ("Last 7 days", now - timedelta(days=7)),
        ("All time", None),
    ]
    for label, since in windows:
        summary = await store.summary(since=since)
        _render_usage_summary(console, label, summary)
        console.print()
    return 0


def _render_usage_summary(console, title: str, summary) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if summary.calls == 0:
        console.print(
            Panel(
                Text(f"(no LLM calls recorded for: {title.lower()})", style="dim"),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        return

    header = Text()
    header.append(f"{summary.calls} call", style="bold")
    header.append(f"{'s' if summary.calls != 1 else ''} · ", style="bold")
    header.append(
        f"{summary.input_tokens:,} in / {summary.output_tokens:,} out tokens · ",
        style="dim",
    )
    header.append(f"${summary.cost_usd:.4f}", style="bold yellow")

    by_purpose = Table(
        title="by purpose", header_style="dim cyan",
        title_style="dim", show_lines=False, expand=False,
    )
    by_purpose.add_column("purpose", style="cyan")
    by_purpose.add_column("calls", justify="right")
    by_purpose.add_column("input", justify="right")
    by_purpose.add_column("output", justify="right")
    by_purpose.add_column("cost", justify="right")
    for name, row in sorted(summary.by_purpose.items()):
        by_purpose.add_row(
            name, str(row.calls),
            f"{row.input_tokens:,}", f"{row.output_tokens:,}",
            f"${row.cost_usd:.4f}",
        )

    by_model = Table(
        title="by model", header_style="dim cyan",
        title_style="dim", show_lines=False, expand=False,
    )
    by_model.add_column("model", style="cyan")
    by_model.add_column("calls", justify="right")
    by_model.add_column("input", justify="right")
    by_model.add_column("output", justify="right")
    by_model.add_column("cost", justify="right")
    for name, row in sorted(summary.by_model.items()):
        by_model.add_row(
            name, str(row.calls),
            f"{row.input_tokens:,}", f"{row.output_tokens:,}",
            f"${row.cost_usd:.4f}",
        )

    from rich.console import Group
    console.print(Panel(
        Group(header, by_purpose, by_model),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))


async def _render_usage_inline(console) -> None:
    """REPL `usage` command — quick session+today snapshot."""
    from datetime import datetime, timezone

    from ..usage import UsageStore

    store = UsageStore(paths.usage_db())
    today_since = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    today = await store.summary(since=today_since)
    all_time = await store.summary()
    _render_usage_summary(console, "Today", today)
    _render_usage_summary(console, "All time", all_time)


# ---------------------------------------------------------------------------
# Observability: `ambi logs` (recent turns) + `ambi status` (health metrics)
# ---------------------------------------------------------------------------


def cmd_logs(args: argparse.Namespace) -> int:
    """Show the most recent agent turns from the telemetry store."""
    return asyncio.run(_run_logs(args))


async def _run_logs(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from ..observability import TelemetryStore

    limit = getattr(args, "limit", None) or 20
    rows = await TelemetryStore(paths.telemetry_db()).recent(limit=limit)
    console = Console()
    if not rows:
        console.print("[dim](no turns recorded yet — run a chat or the daemon)[/dim]")
        return 0

    table = Table(title=f"Last {len(rows)} turns", header_style="bold cyan")
    for col, just in [
        ("when", "left"), ("id", "left"), ("trigger", "left"),
        ("outcome", "center"), ("tools", "right"), ("tok in/out", "right"),
        ("cost", "right"), ("ms", "right"),
    ]:
        table.add_column(col, justify=just)
    for r in rows:
        outcome = r["outcome"]
        colour = {"ok": "green", "error": "red", "max_turns": "yellow"}.get(outcome, "white")
        when = (r["created_at"] or "")[5:19]  # MM-DD HH:MM:SS
        flag = " ⚑" if r["sensegate_flagged"] else ""
        deny = f" ⛔{r['warden_denials']}" if r["warden_denials"] else ""
        table.add_row(
            when, r["turn_id"], r["trigger"],
            f"[{colour}]{outcome}[/{colour}]{flag}{deny}",
            str(r["num_tool_calls"]),
            f"{r['input_tokens']}/{r['output_tokens']}",
            f"${r['cost_usd']:.4f}" if r["cost_usd"] else "—",
            str(r["duration_ms"]),
        )
        if r["error"]:
            table.add_row("", "", "", f"[red]{r['error'][:80]}[/red]", "", "", "", "")
    console.print(table)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show aggregate health metrics over recent turns."""
    return asyncio.run(_run_status(args))


async def _run_status(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from ..observability import TelemetryStore

    s = await TelemetryStore(paths.telemetry_db()).summary()
    console = Console()
    if s.turns == 0:
        console.print("[dim](no turns recorded yet)[/dim]")
        return 0

    by_trig = ", ".join(f"{k}={v}" for k, v in sorted(s.by_trigger.items())) or "—"
    body = Text.from_markup(
        f"[bold]{s.turns}[/bold] recent turns · "
        f"errors [bold]{s.errors}[/bold] ({s.error_rate:.0%}) · "
        f"max-turns {s.max_turns_hits}\n"
        f"latency p50 [bold]{s.p50_ms}ms[/bold] · p95 [bold]{s.p95_ms}ms[/bold]\n"
        f"tool calls {s.total_tool_calls} · "
        f"tokens {s.input_tokens}/{s.output_tokens} · "
        f"cost [bold]${s.cost_usd:.4f}[/bold]\n"
        f"by trigger: {by_trig}"
    )
    border = "red" if s.error_rate > 0.2 else "cyan"
    console.print(Panel(body, title="ambi status", border_style=border, padding=(0, 1)))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run behavioral scenarios under evals/scenarios/ and report pass/fail."""
    return asyncio.run(_run_evals(args))


async def _run_evals(args: argparse.Namespace) -> int:
    import os as _os
    from pathlib import Path as _Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from ..evals import apply_scenario_setup, load_scenarios, run_scenario
    from .build import build_agent

    _load_runtime_env()
    paths.ensure_tree()
    from ..observability import setup_logging
    setup_logging(log_dir=paths.logs_dir(), stderr=True)

    # Default to repo-relative evals/scenarios; allow override.
    scenarios_dir = _Path(args.path) if args.path else _Path("evals/scenarios")
    if not scenarios_dir.exists():
        print(f"error: scenarios path '{scenarios_dir}' not found", file=sys.stderr)
        return 2

    scenarios = load_scenarios(scenarios_dir)
    if not scenarios:
        print(f"(no scenarios under {scenarios_dir})")
        return 0

    console = Console()
    console.print(
        Panel(
            Text.from_markup(
                f"Running [bold]{len(scenarios)}[/bold] scenario(s) from {scenarios_dir}"
            ),
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # Allow up to 2 retries per scenario on transient flakes (Gemini
    # sometimes emits an empty stream). Real regressions still fail
    # because they fail on every attempt.
    max_attempts = int(os.environ.get("AMBI_EVAL_MAX_ATTEMPTS", "2"))

    results = []
    for scenario in scenarios:
        result = None
        for attempt in range(1, max_attempts + 1):
            with apply_scenario_setup(scenario):
                agent = build_agent(
                    extra_tools=[], with_hippocamp=False, task_store=None,
                )
                try:
                    result = await run_scenario(scenario, agent)
                except Exception as e:
                    from ..evals import ScenarioResult
                    result = ScenarioResult(
                        scenario=scenario, response_text="",
                        tools_called=[], input_tokens=0, output_tokens=0,
                        cost_usd=0.0, error=f"{type(e).__name__}: {e}",
                        assertion_results=[],
                    )
            # Pass or has visible content → done. Empty replies on first
            # attempt are usually flake — try again.
            if result.passed or result.response_text.strip() or attempt == max_attempts:
                break
        results.append(result)
        _print_scenario_result(console, result)

    # Summary table.
    table = Table(
        title="Eval summary", header_style="bold cyan",
        title_style="bold", show_lines=False,
    )
    table.add_column("scenario", style="cyan")
    table.add_column("status", justify="center")
    table.add_column("pass/total", justify="right")
    table.add_column("tokens (in/out)", justify="right")
    table.add_column("cost", justify="right")
    pass_n = 0
    tot_in = tot_out = 0
    tot_cost = 0.0
    for r in results:
        passed = sum(1 for a in r.assertion_results if a.passed)
        total = len(r.assertion_results)
        if r.error:
            status = "[red]ERROR[/red]"
        elif r.passed:
            status = "[green]PASS[/green]"
            pass_n += 1
        else:
            status = "[yellow]FAIL[/yellow]"
        tot_in += r.input_tokens
        tot_out += r.output_tokens
        tot_cost += r.cost_usd
        tokens = f"{r.input_tokens}/{r.output_tokens}" if (r.input_tokens or r.output_tokens) else "—"
        cost = f"${r.cost_usd:.4f}" if r.cost_usd else "—"
        table.add_row(r.scenario.name, status, f"{passed}/{total}", tokens, cost)
    console.print(table)
    console.print(
        f"\n[bold]{pass_n}/{len(results)}[/bold] scenarios passed"
        f" · [dim]{tot_in}/{tot_out} tokens · ${tot_cost:.4f} total[/dim]"
    )
    return 0 if pass_n == len(results) else 1


def _print_scenario_result(console, result) -> None:
    from rich.panel import Panel
    from rich.text import Text

    border = "green" if result.passed else ("red" if result.error else "yellow")
    head = Text()
    head.append(result.scenario.name, style="bold")
    if result.scenario.description:
        head.append(f"\n[dim]{result.scenario.description.strip()}[/dim]")
    head.append(f"\n\n[bold]input:[/bold] {result.scenario.input}\n")
    if result.error:
        head.append(f"\n[red]ERROR: {result.error}[/red]")
    else:
        head.append(
            f"\n[dim]tools called:[/dim] "
            f"{', '.join(result.tools_called) or '(none)'}\n"
        )
        head.append(
            f"[dim]reply:[/dim] {result.response_text[:200]}"
            f"{'…' if len(result.response_text) > 200 else ''}\n"
        )
        if result.input_tokens or result.output_tokens or result.cost_usd:
            head.append(
                f"[dim]usage:[/dim] {result.input_tokens} in / "
                f"{result.output_tokens} out · ${result.cost_usd:.4f}\n"
            )
        head.append("\n")
        for a in result.assertion_results:
            tick = "✓" if a.passed else "✗"
            colour = "green" if a.passed else "red"
            val = f": {a.assertion.value}" if a.assertion.value else ""
            head.append(
                f"  [{colour}]{tick}[/{colour}] {a.assertion.type}{val}"
            )
            if not a.passed and a.detail:
                head.append(f"   [dim]→ {a.detail}[/dim]")
            head.append("\n")

    console.print(Panel(Text.from_markup(str(head)), border_style=border, padding=(0, 1)))


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
    sub.add_parser("usage", help="show token + cost summary").set_defaults(func=cmd_usage)
    logs_parser = sub.add_parser("logs", help="show recent agent turns (telemetry)")
    logs_parser.add_argument("-n", "--limit", type=int, default=20,
                             help="how many recent turns to show (default 20)")
    logs_parser.set_defaults(func=cmd_logs)
    sub.add_parser("status", help="aggregate health metrics over recent turns").set_defaults(func=cmd_status)
    eval_parser = sub.add_parser("eval", help="run behavioral scenarios in evals/")
    eval_parser.add_argument("path", nargs="?", default=None,
                             help="scenario file or directory (default: evals/scenarios)")
    eval_parser.set_defaults(func=cmd_eval)
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
