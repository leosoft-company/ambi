"""Unit tests for the bits of the Telegram transport that don't need a bot."""

from types import SimpleNamespace

from datetime import datetime, timedelta, timezone

from ambi.scheduler import ScheduledTask
from ambi.transports.telegram import (
    _describe_cron,
    _format_when,
    _format_when_ago,
    _get_reply_context,
    format_scheduled_list,
    split_message,
)


# ---------- split_message ----------


def test_split_short_text_is_one_chunk():
    assert split_message("hello") == ["hello"]


def test_split_respects_paragraph_boundaries():
    a = "x" * 2000
    b = "y" * 2000
    c = "z" * 2000
    text = f"{a}\n\n{b}\n\n{c}"
    chunks = split_message(text, max_len=4096)
    # Should split at the \n\n between b and c, not mid-paragraph.
    assert len(chunks) == 2
    assert chunks[0].endswith("y" * 2000)
    assert chunks[1] == "z" * 2000


def test_split_falls_back_to_hard_cut_when_no_boundary():
    text = "x" * 5000  # single run, no whitespace
    chunks = split_message(text, max_len=1000)
    assert len(chunks) == 5
    assert all(len(c) <= 1000 for c in chunks)


def test_split_word_boundary():
    text = "word " * 1000  # 5000 chars with spaces
    chunks = split_message(text, max_len=500)
    # Every chunk should end at a word boundary, not mid-word.
    for c in chunks[:-1]:
        assert not c.endswith("word"[:3])  # not in the middle of "word"


# ---------- _get_reply_context ----------


def _msg(**kwargs):
    return SimpleNamespace(**kwargs)


def test_no_reply_returns_none():
    msg = _msg(reply_to_message=None)
    assert _get_reply_context(msg) is None


def test_reply_to_user_text():
    msg = _msg(reply_to_message=_msg(
        from_user=SimpleNamespace(is_bot=False),
        text="my earlier message",
        caption=None,
    ))
    ctx = _get_reply_context(msg)
    assert ctx == "[Replying to yourself]: my earlier message"


def test_reply_to_bot_text():
    msg = _msg(reply_to_message=_msg(
        from_user=SimpleNamespace(is_bot=True),
        text="my prior answer",
        caption=None,
    ))
    ctx = _get_reply_context(msg)
    assert ctx == "[Replying to agent]: my prior answer"


def test_reply_to_photo_no_text():
    msg = _msg(reply_to_message=_msg(
        from_user=SimpleNamespace(is_bot=False),
        text=None,
        caption=None,
        photo=[object()],
        voice=None,
        audio=None,
        document=None,
    ))
    assert _get_reply_context(msg) == "[Replying to yourself's photo]"


def test_long_text_gets_truncated():
    long = "a" * 5000
    msg = _msg(reply_to_message=_msg(
        from_user=SimpleNamespace(is_bot=False),
        text=long,
        caption=None,
    ))
    ctx = _get_reply_context(msg)
    assert ctx.endswith("...")
    assert len(ctx) < 1100  # truncated, not 5000+


# ---------- format_scheduled_list ----------


FIXED_NOW = datetime(2026, 5, 29, 8, 0, 0, tzinfo=timezone.utc)


def _task(
    id="abc12345",
    prompt="do it",
    run_at=None,
    cron=None,
    status="pending",
    created_at=None,
):
    return ScheduledTask(
        id=id,
        prompt=prompt,
        run_at=run_at or datetime(2026, 5, 29, 9, 0, tzinfo=timezone.utc),
        cron=cron,
        status=status,
        last_run_at=None,
        last_result=None,
        run_count=0,
        created_at=created_at,
    )


def test_format_empty_returns_marker():
    assert format_scheduled_list([]) == "(no scheduled tasks)"


def test_format_header_shows_count():
    out = format_scheduled_list([_task(), _task(id="b")], now=FIXED_NOW)
    assert out.startswith("Pending (2):")


def test_format_uses_relative_time_for_one_shot():
    t = _task(run_at=FIXED_NOW + timedelta(minutes=59))
    out = format_scheduled_list([t], now=FIXED_NOW)
    assert "in 59 min" in out


def test_format_does_not_show_internal_id():
    """User-facing list uses numbered prefixes, not raw IDs."""
    out = format_scheduled_list([_task(id="11d43c58")], now=FIXED_NOW)
    assert "11d43c58" not in out
    assert "1." in out


def test_format_numbers_each_task():
    out = format_scheduled_list(
        [_task(id="a"), _task(id="b"), _task(id="c")], now=FIXED_NOW,
    )
    assert "1." in out
    assert "2." in out
    assert "3." in out


def test_format_decodes_daily_cron():
    out = format_scheduled_list([_task(cron="0 7 * * *")], now=FIXED_NOW)
    assert "repeats daily at 07:00 UTC" in out


def test_format_includes_added_when_created_at_set():
    out = format_scheduled_list(
        [_task(created_at=FIXED_NOW - timedelta(hours=2))],
        now=FIXED_NOW,
    )
    assert "added 2h ago" in out


def test_format_completed_shows_status_tag():
    out = format_scheduled_list(
        [_task(status="completed")], include_done=True, now=FIXED_NOW,
    )
    assert "[completed]" in out
    assert out.startswith("All tasks (1):")


def test_format_long_prompt_truncated():
    long = "x" * 500
    out = format_scheduled_list([_task(prompt=long)], now=FIXED_NOW)
    assert "…" in out
    assert "x" * 500 not in out


# ---------- _format_when ----------


def test_format_when_due_now():
    assert _format_when(FIXED_NOW, FIXED_NOW) == "due now"


def test_format_when_minutes_only():
    assert _format_when(FIXED_NOW + timedelta(minutes=5), FIXED_NOW) == "in 5 min"


def test_format_when_hours_no_minutes():
    assert _format_when(FIXED_NOW + timedelta(hours=3), FIXED_NOW) == "in 3h"


def test_format_when_hours_and_minutes():
    assert _format_when(FIXED_NOW + timedelta(hours=3, minutes=15), FIXED_NOW) == "in 3h 15m"


def test_format_when_days():
    assert _format_when(FIXED_NOW + timedelta(days=2), FIXED_NOW) == "in 2d"


def test_format_when_weeks():
    assert _format_when(FIXED_NOW + timedelta(days=21), FIXED_NOW) == "in 3w"


# ---------- _format_when_ago ----------


def test_format_when_ago_minutes():
    assert _format_when_ago(FIXED_NOW - timedelta(minutes=15), FIXED_NOW) == "15 min ago"


def test_format_when_ago_hours():
    assert _format_when_ago(FIXED_NOW - timedelta(hours=5), FIXED_NOW) == "5h ago"


# ---------- _describe_cron ----------


def test_describe_cron_daily():
    assert _describe_cron("0 7 * * *") == "daily at 07:00 UTC"


def test_describe_cron_hourly():
    assert _describe_cron("15 * * * *") == "hourly at :15"


def test_describe_cron_weekly():
    assert _describe_cron("0 9 * * 1") == "weekly on Mon at 09:00 UTC"


def test_describe_cron_unknown_passes_through():
    expr = "*/5 * * * *"
    assert _describe_cron(expr) == expr
