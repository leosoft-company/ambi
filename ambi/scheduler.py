"""Scheduler — fire `agent.chat(prompt)` at scheduled times.

Storage: a `scheduled_tasks` table in the same SQLite file as the session
store. Tasks are either:
    - one-shot:   `run_at` set, `cron` null  -> status becomes 'completed' after fire
    - recurring:  `cron` set                 -> `run_at` advances after each fire

Runner: a single background poll loop. On each tick it fetches due tasks,
fires each through `agent.chat()` (serialized by the Agent's own lock),
and routes the resulting text via the `on_result(task, reply)` callback —
that's where Telegram delivery happens.

Tools: `schedule(prompt, run_at?, cron?)`, `list_scheduled()`, and
`cancel_scheduled(id)` are exposed to the LLM so it can self-schedule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import aiosqlite
from croniter import croniter

from .tool import Tool
from .types import ToolDef

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id          TEXT PRIMARY KEY,
    prompt      TEXT NOT NULL,
    run_at      TEXT NOT NULL,
    cron        TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    last_run_at TEXT,
    last_result TEXT,
    run_count   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_status_runat
    ON scheduled_tasks(status, run_at);
"""


@dataclass
class ScheduledTask:
    id: str
    prompt: str
    run_at: datetime
    cron: str | None
    status: str
    last_run_at: datetime | None
    last_result: str | None
    run_count: int
    created_at: datetime | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    # SQLite returns naive timestamps for our `datetime('now')` defaults; we
    # store ISO strings on writes. Normalize to UTC-aware on read.
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_task(row) -> ScheduledTask:
    return ScheduledTask(
        id=row[0],
        prompt=row[1],
        run_at=_parse_dt(row[2]),
        cron=row[3],
        status=row[4],
        last_run_at=_parse_dt(row[5]),
        last_result=row[6],
        run_count=row[7],
        created_at=_parse_dt(row[8]) if len(row) > 8 else None,
    )


_SELECT_COLS = (
    "id, prompt, run_at, cron, status, last_run_at, last_result, run_count, created_at"
)


class TaskStore:
    """SQLite-backed store for scheduled tasks. Same DB pattern as SqliteStore."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialized = False

    async def _ensure(self) -> None:
        if self._initialized:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA journal_mode=WAL")  # concurrent reads/writes
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def create(
        self,
        prompt: str,
        run_at: datetime,
        cron: str | None = None,
    ) -> ScheduledTask:
        await self._ensure()
        task_id = secrets.token_hex(4)  # 8 hex chars; 4B combos — plenty per-user
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO scheduled_tasks (id, prompt, run_at, cron) "
                "VALUES (?, ?, ?, ?)",
                (task_id, prompt, run_at.isoformat(), cron),
            )
            await db.commit()
        return ScheduledTask(
            id=task_id, prompt=prompt, run_at=run_at, cron=cron,
            status="pending", last_run_at=None, last_result=None, run_count=0,
        )

    async def get(self, task_id: str) -> ScheduledTask | None:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                f"SELECT {_SELECT_COLS} FROM scheduled_tasks WHERE id = ?",
                (task_id,),
            )
            row = await cur.fetchone()
        return _row_to_task(row) if row else None

    async def list(self, include_done: bool = False) -> list[ScheduledTask]:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            if include_done:
                cur = await db.execute(
                    f"SELECT {_SELECT_COLS} FROM scheduled_tasks "
                    "ORDER BY run_at ASC"
                )
            else:
                cur = await db.execute(
                    f"SELECT {_SELECT_COLS} FROM scheduled_tasks "
                    "WHERE status = 'pending' ORDER BY run_at ASC"
                )
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

    async def due_before(self, when: datetime) -> list[ScheduledTask]:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                f"SELECT {_SELECT_COLS} FROM scheduled_tasks "
                "WHERE status = 'pending' AND run_at <= ? ORDER BY run_at ASC",
                (when.isoformat(),),
            )
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

    async def mark_completed(
        self,
        task_id: str,
        ran_at: datetime,
        result: str,
        next_run_at: datetime | None,
    ) -> None:
        """If next_run_at is None the task is done; else it's recurring."""
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            if next_run_at is None:
                await db.execute(
                    "UPDATE scheduled_tasks SET status='completed', "
                    "last_run_at=?, last_result=?, run_count=run_count+1 "
                    "WHERE id=?",
                    (ran_at.isoformat(), result[:2000], task_id),
                )
            else:
                await db.execute(
                    "UPDATE scheduled_tasks SET last_run_at=?, last_result=?, "
                    "run_count=run_count+1, run_at=? WHERE id=?",
                    (ran_at.isoformat(), result[:2000], next_run_at.isoformat(), task_id),
                )
            await db.commit()

    async def cancel(self, task_id: str) -> bool:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "UPDATE scheduled_tasks SET status='cancelled' "
                "WHERE id=? AND status='pending'",
                (task_id,),
            )
            await db.commit()
            return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


ResultHandler = Callable[[ScheduledTask, str], Awaitable[None]]


class Scheduler:
    """Polls the TaskStore on a fixed interval and fires due tasks."""

    def __init__(
        self,
        store: TaskStore,
        agent,  # ambi.Agent — runtime type to avoid import cycle
        on_result: ResultHandler | None = None,
        check_interval: float = 30.0,
        max_per_tick: int = 5,
    ):
        self.store = store
        self.agent = agent
        self.on_result = on_result
        self.check_interval = check_interval
        # Cap how many due tasks fire in a single tick. After downtime,
        # due_before() can return a large backlog; firing all at once means a
        # burst of (serialized, LLM-cost) chat() calls. The cap drains the
        # backlog gradually over successive ticks instead of stampeding.
        self.max_per_tick = max_per_tick
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        log.info("scheduler_started interval=%ss", self.check_interval)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        log.info("scheduler_stopped")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                log.exception("scheduler_tick_error")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.check_interval,
                )
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        now = _now_utc()
        due = await self.store.due_before(now)
        if len(due) > self.max_per_tick:
            log.warning(
                "scheduler_backlog due=%d firing=%d (rest next tick)",
                len(due), self.max_per_tick,
            )
            due = due[: self.max_per_tick]
        for task in due:
            await self._fire(task)

    async def _fire(self, task: ScheduledTask) -> None:
        from .observability import trigger

        ran_at = _now_utc()
        try:
            with trigger("scheduled"):
                reply = await self.agent.chat(task.prompt)
        except Exception as e:
            log.exception("scheduler_fire_failed task=%s", task.id)
            reply = f"Error executing scheduled task: {type(e).__name__}: {e}"

        next_run = _next_cron_fire(task.cron, ran_at) if task.cron else None
        await self.store.mark_completed(task.id, ran_at, reply, next_run)

        if self.on_result is not None:
            try:
                await self.on_result(task, reply)
            except Exception:
                log.exception("scheduler_on_result_failed task=%s", task.id)


def _next_cron_fire(cron_expr: str, after: datetime) -> datetime:
    it = croniter(cron_expr, after)
    nxt = it.get_next(datetime)
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=timezone.utc)
    return nxt.replace(microsecond=0)


# ---------------------------------------------------------------------------
# LLM-facing tools
# ---------------------------------------------------------------------------


def _validate_run_at(run_at_str: str) -> datetime:
    """Parse ISO 8601. Returns UTC-aware datetime; raises ValueError."""
    dt = datetime.fromisoformat(run_at_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def make_scheduler_tools(store: TaskStore) -> list[Tool]:
    """Build the schedule / list_scheduled / cancel_scheduled tools."""

    async def schedule_handler(args: dict) -> str:
        prompt = args.get("prompt")
        run_at_str = args.get("run_at")
        cron_expr = args.get("cron")

        if not prompt or not isinstance(prompt, str):
            return "Error: 'prompt' (string) is required."
        if not run_at_str and not cron_expr:
            return "Error: provide either 'run_at' (ISO datetime) or 'cron' (cron expression)."

        if cron_expr:
            try:
                croniter(cron_expr)
            except Exception as e:
                return f"Error: invalid cron expression: {e}"
            try:
                run_at = (
                    _validate_run_at(run_at_str)
                    if run_at_str
                    else _next_cron_fire(cron_expr, _now_utc())
                )
            except ValueError as e:
                return f"Error: invalid run_at: {e}"
        else:
            try:
                run_at = _validate_run_at(run_at_str)
            except ValueError as e:
                return f"Error: invalid run_at (need ISO 8601): {e}"

        if run_at < _now_utc():
            return f"Error: run_at {run_at.isoformat()} is in the past."

        task = await store.create(prompt=prompt, run_at=run_at, cron=cron_expr)
        kind = "recurring" if cron_expr else "one-shot"
        return (
            f"Scheduled {kind} task {task.id}: '{prompt[:80]}' "
            f"first fire at {task.run_at.isoformat()}"
        )

    async def list_handler(args: dict) -> str:
        include_done = bool(args.get("include_done", False))
        tasks = await store.list(include_done=include_done)
        if not tasks:
            return "(no scheduled tasks)"
        lines = []
        for t in tasks:
            tag = f"[{t.status}]"
            cron_tag = f" cron={t.cron}" if t.cron else ""
            lines.append(
                f"{t.id} {tag} run_at={t.run_at.isoformat()}{cron_tag} "
                f"runs={t.run_count}\n  prompt: {t.prompt[:160]}"
            )
        return "\n".join(lines)

    async def cancel_handler(args: dict) -> str:
        task_id = args.get("id")
        if not task_id:
            return "Error: 'id' is required."
        ok = await store.cancel(task_id)
        return f"Cancelled {task_id}." if ok else f"No pending task with id {task_id}."

    schedule_tool = Tool(
        definition=ToolDef(
            name="schedule",
            description=(
                "Schedule the agent to run a prompt at a future time. The "
                "prompt will be executed as a regular chat turn (with all "
                "tools available) and the result will be delivered to the "
                "user. Provide either run_at (ISO 8601 datetime in UTC) for "
                "a one-shot, or cron (5-field cron expression) for a "
                "recurring task. If both are given, run_at is the first "
                "fire; subsequent fires follow the cron."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The instruction to run at fire time, e.g. 'Summarize my last 24h of GitHub activity'.",
                    },
                    "run_at": {
                        "type": "string",
                        "description": "ISO 8601 timestamp (UTC) for one-shot or first fire of a recurring task.",
                    },
                    "cron": {
                        "type": "string",
                        "description": "5-field cron expression, e.g. '0 7 * * *' for 07:00 daily UTC.",
                    },
                },
                "required": ["prompt"],
            },
        ),
        handler=schedule_handler,
        kind="write",
    )

    list_tool = Tool(
        definition=ToolDef(
            name="list_scheduled",
            description="List scheduled tasks. By default only pending; set include_done=true for the full history.",
            input_schema={
                "type": "object",
                "properties": {
                    "include_done": {
                        "type": "boolean",
                        "description": "Include completed/cancelled tasks.",
                    },
                },
                "required": [],
            },
        ),
        handler=list_handler,
        kind="read",
    )

    cancel_tool = Tool(
        definition=ToolDef(
            name="cancel_scheduled",
            description="Cancel a pending scheduled task by id.",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Task id returned from schedule()."},
                },
                "required": ["id"],
            },
        ),
        handler=cancel_handler,
        kind="write",
    )

    return [schedule_tool, list_tool, cancel_tool]
