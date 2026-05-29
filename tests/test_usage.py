"""Tests for ambi/usage.py — pricing, store, tracking wrapper."""

from datetime import datetime, timedelta, timezone

import pytest

from ambi.types import CompletionResult, Message, StreamEnd, TextBlock, TextChunk
from ambi.usage import (
    PRICING,
    TrackingProvider,
    UsageStore,
    compute_cost,
    current_purpose,
    purpose,
)


# ---------- pricing ----------


def test_compute_cost_known_model():
    # 1M input + 1M output on flash = $0.075 + $0.30 = $0.375
    cost = compute_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert abs(cost - 0.375) < 1e-9


def test_compute_cost_unknown_model_is_zero():
    assert compute_cost("ollama:llama3", 1_000_000, 1_000_000) == 0.0


def test_pricing_has_canonical_gemini_models():
    for name in ("gemini-2.5-flash", "gemini-2.5-pro"):
        assert name in PRICING


# ---------- UsageStore ----------


async def test_record_and_summary_roundtrip(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    await store.record("s1", "gemini-2.5-flash", "chat", 100, 50)
    await store.record("s1", "gemini-2.5-flash", "sensegate", 200, 30)

    summary = await store.summary()
    assert summary.calls == 2
    assert summary.input_tokens == 300
    assert summary.output_tokens == 80
    assert summary.cost_usd > 0
    assert "chat" in summary.by_purpose
    assert summary.by_purpose["chat"].calls == 1
    assert summary.by_purpose["sensegate"].calls == 1
    assert "gemini-2.5-flash" in summary.by_model


async def test_summary_filters_by_session(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    await store.record("alpha", "gemini-2.5-flash", "chat", 100, 50)
    await store.record("beta", "gemini-2.5-flash", "chat", 200, 30)

    alpha = await store.summary(session_id="alpha")
    assert alpha.calls == 1
    assert alpha.input_tokens == 100
    beta = await store.summary(session_id="beta")
    assert beta.calls == 1
    assert beta.input_tokens == 200


async def test_summary_filters_by_since(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    await store.record("s", "gemini-2.5-flash", "chat", 100, 50)
    far_future = datetime.now(timezone.utc) + timedelta(days=365)
    summary = await store.summary(since=far_future)
    assert summary.calls == 0


async def test_unknown_model_records_zero_cost(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    await store.record("s", "ollama:llama3", "chat", 1000, 500)
    summary = await store.summary()
    assert summary.cost_usd == 0.0
    assert summary.input_tokens == 1000


# ---------- TrackingProvider ----------


class _StubProvider:
    def __init__(self, result):
        self.result = result

    async def complete(self, *a, **kw):
        return self.result

    async def stream(self, *a, **kw):
        yield TextChunk(text="hi")
        yield StreamEnd(stop_reason="end_turn", usage=self.result.usage)


async def test_tracking_provider_records_complete(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    inner = _StubProvider(
        CompletionResult(
            content=[TextBlock("hi")], stop_reason="end_turn",
            usage={"input_tokens": 80, "output_tokens": 20, "model": "gemini-2.5-flash"},
        )
    )
    tracking = TrackingProvider(inner=inner, store=store, session_id="s1")

    await tracking.complete([Message("user", [TextBlock("hi")])], [])

    summary = await store.summary()
    assert summary.calls == 1
    assert summary.input_tokens == 80
    assert summary.output_tokens == 20
    assert summary.cost_usd > 0
    assert "chat" in summary.by_purpose  # default purpose


async def test_tracking_provider_uses_current_purpose(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    inner = _StubProvider(
        CompletionResult(
            content=[TextBlock("ok")], stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 5, "model": "gemini-2.5-flash"},
        )
    )
    tracking = TrackingProvider(inner=inner, store=store)

    with purpose("sensegate"):
        await tracking.complete([Message("user", [TextBlock("x")])], [])
    with purpose("compaction"):
        await tracking.complete([Message("user", [TextBlock("x")])], [])

    summary = await store.summary()
    assert "sensegate" in summary.by_purpose
    assert "compaction" in summary.by_purpose
    assert summary.by_purpose["sensegate"].calls == 1
    assert summary.by_purpose["compaction"].calls == 1


async def test_tracking_provider_streams_record_on_end(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    inner = _StubProvider(
        CompletionResult(
            content=[TextBlock("ok")], stop_reason="end_turn",
            usage={"input_tokens": 40, "output_tokens": 15, "model": "gemini-2.5-flash"},
        )
    )
    tracking = TrackingProvider(inner=inner, store=store)

    chunks = []
    async for chunk in tracking.stream([Message("user", [TextBlock("x")])], []):
        chunks.append(chunk)
    summary = await store.summary()
    assert summary.calls == 1
    assert summary.input_tokens == 40


async def test_tracking_provider_silent_on_empty_usage(tmp_path):
    store = UsageStore(tmp_path / "usage.db")
    inner = _StubProvider(
        CompletionResult(
            content=[TextBlock("ok")], stop_reason="end_turn",
            usage={},  # provider reported nothing
        )
    )
    tracking = TrackingProvider(inner=inner, store=store)

    await tracking.complete([Message("user", [TextBlock("x")])], [])
    summary = await store.summary()
    assert summary.calls == 0  # no row recorded


# ---------- purpose contextvar ----------


def test_purpose_default_is_chat():
    assert current_purpose.get() == "chat"


def test_purpose_context_resets_after_block():
    with purpose("compaction"):
        assert current_purpose.get() == "compaction"
    assert current_purpose.get() == "chat"
