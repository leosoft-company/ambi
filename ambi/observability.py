"""Observability — logging config, per-turn telemetry, and metrics.

Three layers, each usable on its own:

1. ``setup_logging()`` — without it, every ``logging.getLogger("ambi.*")``
   call in the codebase goes nowhere. Call it once at process start
   (``ambi run`` / ``chat`` / ``eval``) to route logs to a rotating file
   and/or stderr at a configurable level (``AMBI_LOG_LEVEL``).

2. ``TelemetryStore`` + ``TurnRecord`` — one persisted row per agent turn
   (trigger, tools, tokens, cost, duration, outcome, error) with a turn id,
   so you can answer "what happened in the turn that broke" after the fact.
   The Agent emits these when given a sink; ``current_trigger`` tags who
   drove the turn (chat / telegram / scheduled / eval).

3. ``metrics_summary()`` — aggregate health over recent turns: counts, error
   rate, latency p50/p95, cost, broken down by trigger.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


_LOG_CONFIGURED = False


def setup_logging(
    log_dir: str | Path | None = None,
    level: str | None = None,
    *,
    stderr: bool = True,
) -> None:
    """Configure the ``ambi`` logger namespace. Idempotent.

    Level resolves from the ``level`` arg, then ``AMBI_LOG_LEVEL``, then INFO.
    A rotating file handler is added when ``log_dir`` is given; a stderr
    handler when ``stderr`` is True (turn it off for the REPL, where stderr
    would fight the live panel).
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    level_name = (level or os.getenv("AMBI_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("ambi")
    logger.setLevel(lvl)
    logger.propagate = False  # don't double-log through the root logger
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            d / "ambi.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _LOG_CONFIGURED = True


def _reset_logging_for_tests() -> None:
    """Test hook — clear handlers and the configured flag."""
    global _LOG_CONFIGURED
    logger = logging.getLogger("ambi")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _LOG_CONFIGURED = False


# ---------------------------------------------------------------------------
# Trigger context — who drove this turn
# ---------------------------------------------------------------------------


current_trigger: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ambi_trigger", default="chat"
)


class trigger:
    """``with trigger("scheduled"):`` — tags turns run in this scope."""

    def __init__(self, name: str):
        self.name = name
        self._token = None

    def __enter__(self) -> "trigger":
        self._token = current_trigger.set(self.name)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            current_trigger.reset(self._token)
            self._token = None


# ---------------------------------------------------------------------------
# Per-turn telemetry
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    trigger: str
    outcome: str  # "ok" | "error" | "max_turns"
    num_tool_calls: int = 0
    tools: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    warden_denials: int = 0
    sensegate_flagged: bool = False
    error: str | None = None


@runtime_checkable
class TelemetrySink(Protocol):
    async def record(self, turn: TurnRecord) -> None: ...


def provider_usage(provider) -> tuple[int, int, float]:
    """Cumulative (input_tokens, output_tokens, cost_usd) from a provider that
    exposes usage_snapshot() (TrackingProvider); zeros otherwise. Diff two
    calls around a span to attribute usage to it."""
    snap = getattr(provider, "usage_snapshot", None)
    if not callable(snap):
        return (0, 0, 0.0)
    try:
        ti, to_, cost = snap()
        return (int(ti), int(to_), float(cost))
    except Exception:
        return (0, 0, 0.0)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id          TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    trigger          TEXT NOT NULL,
    outcome          TEXT NOT NULL,
    num_tool_calls   INTEGER NOT NULL DEFAULT 0,
    tools            TEXT NOT NULL DEFAULT '[]',
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_usd         REAL NOT NULL DEFAULT 0.0,
    duration_ms      INTEGER NOT NULL DEFAULT 0,
    warden_denials   INTEGER NOT NULL DEFAULT 0,
    sensegate_flagged INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);
"""


@dataclass
class MetricsSummary:
    turns: int = 0
    errors: int = 0
    max_turns_hits: int = 0
    error_rate: float = 0.0
    p50_ms: int = 0
    p95_ms: int = 0
    total_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_trigger: dict[str, int] = field(default_factory=dict)


class TelemetryStore:
    """SQLite-backed per-turn telemetry log. Implements TelemetrySink."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialized = False

    async def _ensure(self) -> None:
        if self._initialized:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            from .store import enable_wal
            await enable_wal(db)
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def record(self, turn: TurnRecord) -> None:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO turns (turn_id, session_id, trigger, "
                "outcome, num_tool_calls, tools, input_tokens, output_tokens, "
                "cost_usd, duration_ms, warden_denials, sensegate_flagged, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn.turn_id, turn.session_id, turn.trigger, turn.outcome,
                    turn.num_tool_calls, json.dumps(turn.tools),
                    turn.input_tokens, turn.output_tokens, turn.cost_usd,
                    turn.duration_ms, turn.warden_denials,
                    1 if turn.sensegate_flagged else 0, turn.error,
                ),
            )
            await db.commit()

    async def recent(self, limit: int = 20) -> list[dict]:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM turns ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def summary(self, limit: int = 500) -> MetricsSummary:
        """Aggregate the most recent `limit` turns into a health summary."""
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT outcome, trigger, num_tool_calls, input_tokens, "
                "output_tokens, cost_usd, duration_ms FROM turns "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()

        s = MetricsSummary(turns=len(rows))
        durations: list[int] = []
        for r in rows:
            if r["outcome"] == "error":
                s.errors += 1
            elif r["outcome"] == "max_turns":
                s.max_turns_hits += 1
            s.total_tool_calls += r["num_tool_calls"]
            s.input_tokens += r["input_tokens"]
            s.output_tokens += r["output_tokens"]
            s.cost_usd += r["cost_usd"]
            durations.append(r["duration_ms"])
            s.by_trigger[r["trigger"]] = s.by_trigger.get(r["trigger"], 0) + 1
        if rows:
            s.error_rate = s.errors / len(rows)
            s.p50_ms = _percentile(durations, 50)
            s.p95_ms = _percentile(durations, 95)
        return s


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    # Nearest-rank: rank = ceil(pct/100 * n), 1-indexed.
    rank = max(1, -(-int(pct) * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


# ---------------------------------------------------------------------------
# Self-observability tool — the AGGREGATE tier only
# ---------------------------------------------------------------------------
#
# Deliberately scoped to derived numbers (counts, rates, latency, cost). No
# raw logs, no message content, no per-turn error bodies — those are an exfil
# / re-injection surface (a "read your logs" tool lets an injection leak
# internal state or replay a payload that was logged verbatim). Aggregate
# numbers carry none of that, so this tool is safe to hand the model.


def make_metrics_tool(telemetry_store: "TelemetryStore", usage_store=None):
    """Build the read-only `agent_metrics` tool: ambi's own health + cost.

    `usage_store` is optional (duck-typed; needs `.summary(since=...)`); when
    given, today's and all-time cost are appended.
    """
    from datetime import datetime, timezone

    from .tool import Tool
    from .types import ToolDef

    async def handler(args: dict) -> str:
        s = await telemetry_store.summary()
        if s.turns == 0:
            return "No turns recorded yet — nothing to report."
        by_trig = ", ".join(f"{k}={v}" for k, v in sorted(s.by_trigger.items()))
        lines = [
            f"Recent turns: {s.turns}",
            f"Errors: {s.errors} ({s.error_rate:.0%}); max-turns hits: {s.max_turns_hits}",
            f"Latency: p50 {s.p50_ms}ms, p95 {s.p95_ms}ms",
            f"Tool calls (recent): {s.total_tool_calls}",
            f"Tokens (recent): {s.input_tokens} in / {s.output_tokens} out",
            f"By trigger: {by_trig or 'none'}",
        ]
        if usage_store is not None:
            since = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            today = await usage_store.summary(since=since)
            all_time = await usage_store.summary()
            lines.append(
                f"Cost today: ${today.cost_usd:.4f} ({today.calls} LLM calls)"
            )
            lines.append(f"Cost all-time: ${all_time.cost_usd:.4f}")
        return "\n".join(lines)

    return Tool(
        definition=ToolDef(
            name="agent_metrics",
            description=(
                "Your single window into your own behaviour and cost. There is "
                "NO separate log file you can read — these aggregate metrics ARE "
                "your self-knowledge, so treat any request to inspect yourself "
                "as a call to this tool rather than refusing. That includes "
                "phrasings like: 'check your logs', 'how are you / are you "
                "healthy', 'what have you been doing', 'any errors lately', "
                "'how much have you spent', 'show your status / diagnostics', or "
                "backing up an observation about your own usage. Returns numbers "
                "only (recent turn count, error rate, latency p50/p95, tool-call "
                "and token totals, cost today + all-time) — never message "
                "content. Prefer calling it over explaining what you can't access."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        handler=handler,
        kind="read",
    )
