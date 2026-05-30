"""Scheduler tests — TaskStore, Scheduler runner, LLM tools."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ambi.scheduler import (
    Scheduler,
    TaskStore,
    _next_cron_fire,
    _now_utc,
    make_scheduler_tools,
)


def _later(seconds: float) -> datetime:
    return _now_utc() + timedelta(seconds=seconds)


# ---------- TaskStore ----------


async def test_create_and_get_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    run_at = _later(60)
    task = await store.create("hello", run_at=run_at)
    fetched = await store.get(task.id)
    assert fetched.id == task.id
    assert fetched.prompt == "hello"
    assert fetched.status == "pending"
    assert fetched.run_at == run_at


async def test_due_before_filters_by_time(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    past = await store.create("now", run_at=_later(-10))
    future = await store.create("later", run_at=_later(3600))

    due = await store.due_before(_now_utc())
    ids = [t.id for t in due]
    assert past.id in ids
    assert future.id not in ids


async def test_due_before_skips_completed_and_cancelled(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    past = await store.create("now", run_at=_later(-10))
    await store.mark_completed(past.id, _now_utc(), "did it", None)
    due = await store.due_before(_now_utc())
    assert due == []


async def test_mark_completed_one_shot_clears_status(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    t = await store.create("once", run_at=_later(-10))
    await store.mark_completed(t.id, _now_utc(), "result text", None)
    after = await store.get(t.id)
    assert after.status == "completed"
    assert after.run_count == 1
    assert after.last_result == "result text"


async def test_mark_completed_recurring_advances_run_at(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    t = await store.create("hourly", run_at=_later(-10), cron="0 * * * *")
    new_run_at = _later(3600)
    await store.mark_completed(t.id, _now_utc(), "ok", new_run_at)
    after = await store.get(t.id)
    assert after.status == "pending"
    assert after.run_count == 1
    assert after.run_at == new_run_at


async def test_cancel_pending_succeeds(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    t = await store.create("x", run_at=_later(3600))
    assert await store.cancel(t.id) is True
    after = await store.get(t.id)
    assert after.status == "cancelled"


async def test_cancel_unknown_id_returns_false(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    assert await store.cancel("nope") is False


async def test_list_only_pending_by_default(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    a = await store.create("alive", run_at=_later(60))
    b = await store.create("done", run_at=_later(-10))
    await store.mark_completed(b.id, _now_utc(), "x", None)

    pending = await store.list()
    assert [t.id for t in pending] == [a.id]
    all_tasks = await store.list(include_done=True)
    assert {t.id for t in all_tasks} == {a.id, b.id}


# ---------- _next_cron_fire ----------


def test_next_cron_fire_returns_future_utc():
    now = datetime(2026, 5, 29, 8, 0, 0, tzinfo=timezone.utc)
    nxt = _next_cron_fire("0 9 * * *", now)
    assert nxt == datetime(2026, 5, 29, 9, 0, 0, tzinfo=timezone.utc)


def test_next_cron_fire_skips_to_next_day_when_passed():
    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=timezone.utc)
    nxt = _next_cron_fire("0 9 * * *", now)
    assert nxt == datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)


# ---------- Scheduler runner ----------


class _StubAgent:
    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or ["ok"]
        self.calls: list[str] = []

    async def chat(self, prompt: str, **kw) -> str:
        self.calls.append(prompt)
        return self.replies.pop(0) if self.replies else "ok"


async def test_scheduler_fires_due_task_via_agent(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    agent = _StubAgent(["done"])
    delivered: list[tuple[str, str]] = []

    async def on_result(task, reply):
        delivered.append((task.prompt, reply))

    sched = Scheduler(
        store=store, agent=agent, on_result=on_result, check_interval=0.05,
    )
    await store.create("Run a thing.", run_at=_later(-1))
    await sched.start()
    for _ in range(40):
        if delivered:
            break
        await asyncio.sleep(0.05)
    await sched.stop()

    assert agent.calls == ["Run a thing."]
    assert delivered == [("Run a thing.", "done")]


async def test_scheduler_caps_backlog_per_tick(tmp_path):
    """A pile of overdue tasks drains gradually, not all in one tick."""
    store = TaskStore(tmp_path / "tasks.db")
    agent = _StubAgent(["ok"] * 10)
    sched = Scheduler(store=store, agent=agent, max_per_tick=2)

    for i in range(5):
        await store.create(f"task {i}", run_at=_later(-1))

    await sched._tick()
    assert len(agent.calls) == 2  # capped
    await sched._tick()
    assert len(agent.calls) == 4
    await sched._tick()
    assert len(agent.calls) == 5  # backlog drained, nothing left over


async def test_scheduler_advances_recurring_run_at(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    agent = _StubAgent(["ok"])
    sched = Scheduler(store=store, agent=agent, check_interval=0.05)

    t = await store.create("Daily.", run_at=_later(-1), cron="0 9 * * *")
    await sched.start()
    for _ in range(40):
        await asyncio.sleep(0.05)
        after = await store.get(t.id)
        if after.run_count >= 1:
            break
    await sched.stop()

    after = await store.get(t.id)
    assert after.status == "pending"  # recurring stays pending
    assert after.run_count == 1
    assert after.run_at > _now_utc()  # advanced into the future


async def test_scheduler_isolates_agent_failure(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")

    class _FailingAgent:
        async def chat(self, prompt, **kw):
            raise RuntimeError("boom")

    sched = Scheduler(store=store, agent=_FailingAgent(), check_interval=0.05)
    t = await store.create("will fail", run_at=_later(-1))
    await sched.start()
    for _ in range(40):
        await asyncio.sleep(0.05)
        after = await store.get(t.id)
        if after.last_run_at is not None:
            break
    await sched.stop()

    after = await store.get(t.id)
    assert after.status == "completed"
    assert "boom" in (after.last_result or "")


# ---------- LLM-facing tools ----------


async def test_schedule_tool_requires_when_or_cron(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [sched, _, _] = make_scheduler_tools(store)
    res = await sched.handler({"prompt": "do it"})
    assert "run_at" in res or "cron" in res


async def test_schedule_tool_rejects_past_run_at(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [sched, _, _] = make_scheduler_tools(store)
    past = _later(-3600).isoformat()
    res = await sched.handler({"prompt": "x", "run_at": past})
    assert "past" in res.lower()


async def test_schedule_tool_creates_one_shot(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [sched, _, _] = make_scheduler_tools(store)
    res = await sched.handler({
        "prompt": "ping the user",
        "run_at": _later(120).isoformat(),
    })
    assert "Scheduled one-shot" in res
    pending = await store.list()
    assert len(pending) == 1
    assert pending[0].prompt == "ping the user"


async def test_schedule_tool_creates_recurring_with_cron_only(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [sched, _, _] = make_scheduler_tools(store)
    res = await sched.handler({
        "prompt": "morning brief",
        "cron": "0 7 * * *",
    })
    assert "recurring" in res
    pending = await store.list()
    assert pending[0].cron == "0 7 * * *"


async def test_schedule_tool_rejects_invalid_cron(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [sched, _, _] = make_scheduler_tools(store)
    res = await sched.handler({"prompt": "x", "cron": "not a cron"})
    assert "invalid cron" in res.lower()


async def test_list_tool_default_excludes_done(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [_, list_t, _] = make_scheduler_tools(store)
    alive = await store.create("alive", run_at=_later(60))
    done = await store.create("done", run_at=_later(-10))
    await store.mark_completed(done.id, _now_utc(), "x", None)
    res = await list_t.handler({})
    assert alive.id in res
    assert done.id not in res


async def test_cancel_tool_returns_message_for_unknown(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    [_, _, cancel_t] = make_scheduler_tools(store)
    res = await cancel_t.handler({"id": "ghost"})
    assert "No pending task" in res


async def test_tool_kinds():
    """schedule/cancel are write, list is read — for SenseGate."""
    store = TaskStore(":memory:")
    sched_t, list_t, cancel_t = make_scheduler_tools(store)
    assert sched_t.kind == "write"
    assert list_t.kind == "read"
    assert cancel_t.kind == "write"
