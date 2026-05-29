"""Telegram transport — pipes text messages into Agent.chat().

Single shared Agent: every authorized user converses with the same brain
(matches ambi-core's continuous-session design). Concurrent messages are
serialized through an asyncio.Lock so history never interleaves.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..loop import Agent
from ..scheduler import ScheduledTask, TaskStore

log = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096
TYPING_REFRESH_SECONDS = 4


def _get_reply_context(message) -> str | None:
    """Extract a short prefix describing the message being replied to."""
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    from_user = getattr(reply, "from_user", None)
    is_bot = getattr(from_user, "is_bot", False) if from_user else False
    sender = "agent" if is_bot else "yourself"

    quoted = reply.text or reply.caption or ""
    if quoted:
        max_len = 2000 if is_bot else 1000
        if len(quoted) > max_len:
            quoted = quoted[:max_len] + "..."
        return f"[Replying to {sender}]: {quoted}"

    if getattr(reply, "photo", None):
        return f"[Replying to {sender}'s photo]"
    if getattr(reply, "voice", None) or getattr(reply, "audio", None):
        return f"[Replying to {sender}'s voice/audio message]"
    if getattr(reply, "document", None):
        return f"[Replying to {sender}'s document]"
    return None


def split_message(text: str, max_len: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split at paragraph > line > word boundaries before falling back to hard cut."""
    if len(text) <= max_len:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = remaining.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = remaining.rfind(" ", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        out.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        out.append(remaining)
    return out


def format_scheduled_list(
    tasks: list[ScheduledTask],
    include_done: bool = False,
    now: datetime | None = None,
) -> str:
    if not tasks:
        return "(no scheduled tasks)"
    now = now or datetime.now(timezone.utc)
    header = (
        f"All tasks ({len(tasks)}):"
        if include_done
        else f"Pending ({len(tasks)}):"
    )
    lines: list[str] = []
    for i, t in enumerate(tasks, 1):
        when = _format_when(t.run_at, now)
        local = _format_local_clock(t.run_at)
        cron_suffix = (
            f" • repeats {_describe_cron(t.cron)}" if t.cron else ""
        )
        status_tag = (
            f" [{t.status}]" if t.status != "pending" else ""
        )
        prompt = t.prompt.replace("\n", " ").strip()
        if len(prompt) > 140:
            prompt = prompt[:140] + "…"
        added = (
            f" • added {_format_when_ago(t.created_at, now)}"
            if t.created_at
            else ""
        )
        lines.append(
            f"{i}.{status_tag} {when} ({local}){cron_suffix}{added}\n"
            f"   {prompt}"
        )
    return f"{header}\n\n" + "\n\n".join(lines)


def _format_when(run_at: datetime, now: datetime) -> str:
    delta_s = (run_at - now).total_seconds()
    if delta_s <= 0:
        return "due now"
    minutes = int(delta_s // 60)
    if minutes < 1:
        return "any moment"
    if minutes < 60:
        return f"in {minutes} min"
    hours, rem_min = divmod(minutes, 60)
    if hours < 24:
        return f"in {hours}h {rem_min:02d}m" if rem_min else f"in {hours}h"
    days, rem_h = divmod(hours, 24)
    if days < 7:
        return f"in {days}d {rem_h}h" if rem_h else f"in {days}d"
    return f"in {days // 7}w"


def _format_when_ago(when: datetime, now: datetime) -> str:
    delta_s = (now - when).total_seconds()
    if delta_s < 0:
        return "just now"
    minutes = int(delta_s // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours, _ = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    return f"{days // 7}w ago"


def _format_local_clock(run_at: datetime) -> str:
    """Render in the process's local timezone — usually the user's clock."""
    local = run_at.astimezone()
    return local.strftime("%a %H:%M %Z").strip()


def _describe_cron(expr: str) -> str:
    parts = expr.split()
    if len(parts) != 5:
        return expr
    m, h, dom, mo, dow = parts

    def hm(h_, m_) -> str:
        return f"{int(h_):02d}:{int(m_):02d}"

    if dom == "*" and mo == "*" and dow == "*":
        if m.isdigit() and h.isdigit():
            return f"daily at {hm(h, m)} UTC"
        if h == "*" and m.isdigit():
            return f"hourly at :{int(m):02d}"
    if dom == "*" and mo == "*" and dow.isdigit() and m.isdigit() and h.isdigit():
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        idx = int(dow) % 7
        return f"weekly on {days[idx]} at {hm(h, m)} UTC"
    return expr


class TelegramTransport:
    def __init__(
        self,
        agent: Agent,
        bot_token: str,
        allowed_user_ids: set[int] | None = None,
        chat_timeout: float = 180.0,
        task_store: TaskStore | None = None,
    ):
        self.agent = agent
        self.bot_token = bot_token
        self.allowed_user_ids = allowed_user_ids
        self.chat_timeout = chat_timeout
        self.task_store = task_store
        self._app: Application | None = None

    def _is_authorized(self, user_id: int) -> bool:
        if not self.allowed_user_ids:
            return True  # empty/None = allow all (dev mode)
        return user_id in self.allowed_user_ids

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._app = (
            Application.builder()
            .token(self.bot_token)
            .concurrent_updates(True)
            .build()
        )
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("ping", self._cmd_ping))
        if self.task_store is not None:
            self._app.add_handler(CommandHandler("scheduled", self._cmd_scheduled))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        await self._app.initialize()
        await self._app.start()
        # Cancel any lingering getUpdates session from a previously killed
        # instance — otherwise a SIGKILL'd process leaves an active long-poll
        # on Telegram's servers for ~30s and causes Conflict on restart.
        try:
            await self._app.bot.get_updates(offset=-1, timeout=0)
        except Exception:
            pass
        await self._app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message"],
        )
        log.info("telegram_polling_started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        log.info("telegram_polling_stopped")

    # ---------- handlers ----------

    async def _cmd_start(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text(
            "ambi ready. Send a message to chat. /ping to verify I'm alive."
        )

    async def _cmd_ping(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_authorized(update.effective_user.id):
            return
        await update.message.reply_text("pong")

    async def _cmd_scheduled(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_authorized(update.effective_user.id):
            return
        if self.task_store is None:
            await update.message.reply_text("Scheduler not wired.")
            return
        include_done = any(a.lower() == "all" for a in (ctx.args or []))
        tasks = await self.task_store.list(include_done=include_done)
        text = format_scheduled_list(tasks, include_done=include_done)
        await self._send_split(update, text)

    async def _on_message(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return

        text = update.message.text or ""
        reply_ctx = _get_reply_context(update.message)
        if reply_ctx:
            text = f"{reply_ctx}\n\n{text}"

        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(self._typing_loop(ctx, chat_id))
        try:
            reply = await asyncio.wait_for(
                self.agent.chat(text), timeout=self.chat_timeout,
            )
        except asyncio.TimeoutError:
            reply = f"(chat timed out after {self.chat_timeout}s)"
        except Exception as e:
            log.exception("telegram_chat_error")
            reply = f"Error: {type(e).__name__}: {e}"
        finally:
            typing_task.cancel()

        await self._send_split(update, reply)

    async def _typing_loop(self, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        try:
            while True:
                try:
                    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
                except Exception:
                    pass
                await asyncio.sleep(TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _send_split(self, update: Update, text: str) -> None:
        if not text:
            return
        for chunk in split_message(text):
            try:
                await update.message.reply_text(chunk)
            except BadRequest as e:
                log.warning("telegram_send_failed: %s", e)
