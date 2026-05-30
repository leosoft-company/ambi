"""Observability — logging config, telemetry store/metrics, and the loop's
per-turn telemetry emission."""

import logging

import pytest

from ambi.loop import Agent
from ambi.observability import (
    MetricsSummary,
    TelemetryStore,
    TurnRecord,
    _percentile,
    _reset_logging_for_tests,
    current_trigger,
    provider_usage,
    setup_logging,
    trigger,
)
from ambi.tool import ToolKind, ToolRegistry
from ambi.types import (
    CompletionResult,
    TextBlock,
    ToolDef,
    ToolUseBlock,
)
from ambi.usage import TrackingProvider, UsageStore

from tests.mock_provider import MockProvider


# ---------- logging config ----------


def test_setup_logging_configures_level_and_handlers(tmp_path):
    _reset_logging_for_tests()
    try:
        setup_logging(log_dir=tmp_path, level="WARNING", stderr=False)
        lg = logging.getLogger("ambi")
        assert lg.level == logging.WARNING
        assert lg.handlers  # at least the file handler
        lg.warning("hello-observability")
        for h in lg.handlers:
            h.flush()
        log_file = tmp_path / "ambi.log"
        assert log_file.exists()
        assert "hello-observability" in log_file.read_text()
    finally:
        _reset_logging_for_tests()


def test_setup_logging_is_idempotent(tmp_path):
    _reset_logging_for_tests()
    try:
        setup_logging(log_dir=tmp_path, stderr=False)
        n = len(logging.getLogger("ambi").handlers)
        setup_logging(log_dir=tmp_path, stderr=False)
        assert len(logging.getLogger("ambi").handlers) == n  # no duplicate handlers
    finally:
        _reset_logging_for_tests()


# ---------- trigger contextvar ----------


def test_trigger_sets_and_restores():
    assert current_trigger.get() == "chat"
    with trigger("telegram"):
        assert current_trigger.get() == "telegram"
    assert current_trigger.get() == "chat"


# ---------- percentile ----------


def test_percentile_nearest_rank():
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 50) == 30
    assert _percentile(vals, 95) == 50
    assert _percentile([], 50) == 0


# ---------- provider_usage ----------


def test_provider_usage_zero_without_tracking():
    assert provider_usage(object()) == (0, 0, 0.0)


def test_provider_usage_reads_tracking_provider():
    tp = TrackingProvider(inner=object(), store=UsageStore(":memory:"))
    tp._acc_input, tp._acc_output, tp._acc_cost = 100, 40, 0.001
    assert provider_usage(tp) == (100, 40, 0.001)


# ---------- TelemetryStore ----------


async def test_telemetry_store_record_and_recent(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    await store.record(TurnRecord(
        turn_id="t1", session_id="s", trigger="chat", outcome="ok",
        num_tool_calls=2, tools=["a", "b"], input_tokens=10, output_tokens=5,
        cost_usd=0.001, duration_ms=42,
    ))
    rows = await store.recent()
    assert len(rows) == 1
    assert rows[0]["turn_id"] == "t1"
    assert rows[0]["num_tool_calls"] == 2


async def test_telemetry_store_summary_aggregates(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry.db")
    await store.record(TurnRecord("t1", "s", "chat", "ok", duration_ms=100))
    await store.record(TurnRecord("t2", "s", "telegram", "error", duration_ms=300,
                                  error="boom"))
    await store.record(TurnRecord("t3", "s", "scheduled", "max_turns", duration_ms=200))
    s = await store.summary()
    assert s.turns == 3
    assert s.errors == 1
    assert s.max_turns_hits == 1
    assert s.error_rate == pytest.approx(1 / 3)
    assert s.by_trigger == {"chat": 1, "telegram": 1, "scheduled": 1}
    assert s.p50_ms in (200, 300)  # nearest-rank over [100,200,300]


# ---------- loop emits telemetry ----------


def _tool(name, handler, kind: ToolKind = "read"):
    from ambi.tool import Tool
    return Tool(
        definition=ToolDef(name=name, description=name,
                           input_schema={"type": "object", "properties": {}, "required": []}),
        handler=handler, kind=kind,
    )


class _CapturingSink:
    def __init__(self):
        self.turns: list[TurnRecord] = []

    async def record(self, turn: TurnRecord) -> None:
        self.turns.append(turn)


async def test_loop_emits_ok_turn_record():
    tools = ToolRegistry()

    async def add(args):
        return "3"

    tools.register(_tool("add", add))
    sink = _CapturingSink()
    provider = MockProvider([
        CompletionResult(content=[ToolUseBlock(id="t1", name="add", input={})],
                         stop_reason="tool_use"),
        CompletionResult(content=[TextBlock("done")], stop_reason="end_turn"),
    ])
    agent = Agent(provider=provider, tools=tools, system="s", telemetry=sink)
    await agent.chat("go")

    assert len(sink.turns) == 1
    rec = sink.turns[0]
    assert rec.outcome == "ok"
    assert rec.trigger == "chat"
    assert rec.tools == ["add"]
    assert rec.num_tool_calls == 1


async def test_loop_emits_error_turn_record():
    sink = _CapturingSink()

    class Boom:
        async def complete(self, *a, **k):
            raise RuntimeError("down")

        async def stream(self, *a, **k):
            raise RuntimeError("down")
            yield  # generator marker

    agent = Agent(provider=Boom(), tools=ToolRegistry(), system="s", telemetry=sink)
    with pytest.raises(RuntimeError, match="down"):
        await agent.chat("go")

    assert len(sink.turns) == 1
    assert sink.turns[0].outcome == "error"
    assert "down" in sink.turns[0].error


async def test_metrics_tool_reports_aggregate(tmp_path):
    from ambi.observability import make_metrics_tool

    store = TelemetryStore(tmp_path / "telemetry.db")
    await store.record(TurnRecord("t1", "s", "chat", "ok", num_tool_calls=2,
                                  input_tokens=100, output_tokens=20, duration_ms=500))
    await store.record(TurnRecord("t2", "s", "telegram", "error", duration_ms=900,
                                  error="boom"))

    tool = make_metrics_tool(store)  # no usage store → no cost lines
    out = await tool.handler({})
    assert "Recent turns: 2" in out
    assert "Errors: 1" in out
    assert "By trigger:" in out
    assert "chat=1" in out and "telegram=1" in out
    # Aggregate only — no message content / error bodies leak through.
    assert "boom" not in out
    assert tool.kind == "read"


async def test_metrics_tool_includes_cost_with_usage_store(tmp_path):
    from ambi.observability import make_metrics_tool

    tstore = TelemetryStore(tmp_path / "telemetry.db")
    await tstore.record(TurnRecord("t1", "s", "chat", "ok"))
    ustore = UsageStore(tmp_path / "usage.db")
    await ustore.record("s", "gemini-2.5-flash", "chat", 1000, 200)

    tool = make_metrics_tool(tstore, ustore)
    out = await tool.handler({})
    assert "Cost today:" in out
    assert "Cost all-time:" in out


async def test_metrics_tool_empty_is_graceful(tmp_path):
    from ambi.observability import make_metrics_tool

    tool = make_metrics_tool(TelemetryStore(tmp_path / "telemetry.db"))
    out = await tool.handler({})
    assert "No turns recorded" in out


async def test_loop_tags_trigger_from_context():
    sink = _CapturingSink()
    provider = MockProvider([
        CompletionResult(content=[TextBlock("hi")], stop_reason="end_turn"),
    ])
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s", telemetry=sink)
    with trigger("scheduled"):
        await agent.chat("go")
    assert sink.turns[0].trigger == "scheduled"
