"""LLM usage tracking — token counts + USD cost per call, persisted to SQLite.

Design:

- `UsageStore` writes one row per provider call to `~/.ambi/data/usage.db`
  with `(session_id, model, purpose, input_tokens, output_tokens, cost_usd,
  created_at)`.
- `TrackingProvider` wraps any `LLMProvider`; reads a `current_purpose`
  contextvar set by the caller (chat / sensegate / compaction) so each row
  is tagged with what the call was for.
- Pricing table is hard-coded for known Gemini models — unknown models
  record 0 cost so the token totals stay correct.

Why contextvar instead of explicit purpose= arg: every call site (Agent,
verifier, scheduler) eventually calls `provider.complete()`. Threading an
extra arg through `LLMProvider` would force every adapter to learn about
tracking. The contextvar lets the wrapper observe the caller's intent
without changing the seam.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from .provider import LLMProvider
from .types import CompletionResult, Message, ProviderChunk, StreamEnd, ToolDef


# ---------------------------------------------------------------------------
# Purpose context (set by callers, read by TrackingProvider)
# ---------------------------------------------------------------------------


current_purpose: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ambi_purpose", default="chat"
)


class purpose:
    """`async with purpose("compaction"):` — sets the current call's tag."""

    def __init__(self, name: str):
        self.name = name
        self._token = None

    def __enter__(self) -> "purpose":
        self._token = current_purpose.set(self.name)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            current_purpose.reset(self._token)
            self._token = None


# ---------------------------------------------------------------------------
# Pricing — USD per 1M tokens (input, output)
# ---------------------------------------------------------------------------


PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost given a model name and token counts."""
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL DEFAULT 'default',
    model         TEXT NOT NULL,
    purpose       TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_session ON llm_usage(session_id, created_at);
"""


@dataclass
class UsageRow:
    input_tokens: int
    output_tokens: int
    cost_usd: float
    calls: int

    def add(self, ti: int, to_: int, cost: float) -> None:
        self.input_tokens += ti
        self.output_tokens += to_
        self.cost_usd += cost
        self.calls += 1


@dataclass
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_purpose: dict[str, UsageRow] = field(default_factory=dict)
    by_model: dict[str, UsageRow] = field(default_factory=dict)


class UsageStore:
    """Async SQLite-backed usage log."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialized = False

    async def _ensure(self) -> None:
        if self._initialized:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def record(
        self,
        session_id: str,
        model: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        await self._ensure()
        cost = compute_cost(model, input_tokens, output_tokens)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO llm_usage "
                "(session_id, model, purpose, input_tokens, output_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, model, purpose, input_tokens, output_tokens, cost),
            )
            await db.commit()

    async def summary(
        self,
        since: datetime | None = None,
        session_id: str | None = None,
    ) -> UsageSummary:
        await self._ensure()
        clauses: list[str] = []
        params: list = []
        if since is not None:
            clauses.append("created_at >= ?")
            # SQLite's datetime('now') is UTC naive, formatted "YYYY-MM-DD HH:MM:SS"
            params.append(since.strftime("%Y-%m-%d %H:%M:%S"))
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                f"SELECT model, purpose, input_tokens, output_tokens, cost_usd "
                f"FROM llm_usage{where}",
                params,
            )
            rows = await cur.fetchall()

        summary = UsageSummary()
        for model, purp, ti, to_, cost in rows:
            summary.input_tokens += ti
            summary.output_tokens += to_
            summary.cost_usd += cost
            summary.calls += 1
            summary.by_purpose.setdefault(purp, UsageRow(0, 0, 0.0, 0)).add(ti, to_, cost)
            summary.by_model.setdefault(model, UsageRow(0, 0, 0.0, 0)).add(ti, to_, cost)
        return summary


# ---------------------------------------------------------------------------
# Tracking wrapper
# ---------------------------------------------------------------------------


class TrackingProvider:
    """Wraps an LLMProvider, recording token usage of every call.

    Reads `current_purpose` to tag each row — set it with the `purpose(...)`
    context manager around the call site (see ambi/loop.py for examples).
    Unwrapped callers default to "chat".
    """

    def __init__(
        self,
        inner: LLMProvider,
        store: UsageStore,
        session_id: str = "default",
    ):
        self.inner = inner
        self.store = store
        self.session_id = session_id

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> CompletionResult:
        result = await self.inner.complete(
            messages, tools, system=system, max_tokens=max_tokens, **provider_kwargs,
        )
        await self._record(result.usage)
        return result

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
        **provider_kwargs,
    ) -> AsyncIterator[ProviderChunk]:
        async for chunk in self.inner.stream(
            messages, tools, system=system, max_tokens=max_tokens, **provider_kwargs,
        ):
            if isinstance(chunk, StreamEnd):
                await self._record(chunk.usage)
            yield chunk

    async def _record(self, usage: dict) -> None:
        if not usage:
            return
        model = usage.get("model") or "unknown"
        ti = int(usage.get("input_tokens", 0) or 0)
        to_ = int(usage.get("output_tokens", 0) or 0)
        if ti == 0 and to_ == 0:
            return
        try:
            await self.store.record(
                session_id=self.session_id,
                model=model,
                purpose=current_purpose.get(),
                input_tokens=ti,
                output_tokens=to_,
            )
        except Exception:
            # Never let usage tracking interrupt the chat path.
            pass
