"""Warden — pre-execution policy enforcement for tool calls.

Where SenseGate verifies *what happened* after a turn (best-effort LLM
judge), Warden authorizes *what may happen* before it does (deterministic
gate). Different stage, different guarantee.

Composition: a Warden holds an ordered list of Policy objects. Each
policy returns Allow / Deny / RequireConfirmation. First non-allow wins.
Allow-by-default — an empty Warden imposes no constraints.

This module ships four starter policies:

  AllowlistPolicy        — restrict a tool's input to a set of values
  CommandAllowlistPolicy — argv[0]-basename allowlist for run_command
  CostCeilingPolicy      — deny tools once daily USD spend exceeds budget
  QuietHoursPolicy       — deny tools during a local-time window
  ArgvValidatorPolicy    — deny argv patterns for run_command

The starter set is opinionated for personal-AI use. Custom policies just
need to satisfy the Policy protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


Verdict = Literal["allow", "deny", "require_confirmation"]


@dataclass
class PolicyDecision:
    verdict: Verdict
    reason: str = ""
    policy_name: str = ""


@dataclass
class PolicyContext:
    """What a policy gets to inspect before deciding."""

    tool_name: str
    tool_input: dict
    session_id: str = "default"
    now_utc: datetime | None = None


class Policy(Protocol):
    name: str

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision: ...


@dataclass
class AuditEntry:
    at: datetime
    tool_name: str
    verdict: Verdict
    reason: str
    policy_name: str


# ---------------------------------------------------------------------------
# Warden
# ---------------------------------------------------------------------------


class Warden:
    """Authorizes tool calls against a stack of policies.

    Behaviour:
      - Policies evaluated in order; first non-allow verdict wins.
      - Empty policy list = allow everything (no-op Warden).
      - Every decision (including allows) goes to `audit_log` for later
        inspection.
    """

    def __init__(self, policies: list[Policy] | None = None):
        self.policies: list[Policy] = list(policies) if policies else []
        self.audit_log: list[AuditEntry] = []

    def add(self, policy: Policy) -> "Warden":
        self.policies.append(policy)
        return self

    async def authorize(self, ctx: PolicyContext) -> PolicyDecision:
        for policy in self.policies:
            decision = await policy.evaluate(ctx)
            if decision.verdict != "allow":
                self._record(ctx, decision)
                return decision
        decision = PolicyDecision(verdict="allow", policy_name="warden:default")
        self._record(ctx, decision)
        return decision

    def _record(self, ctx: PolicyContext, decision: PolicyDecision) -> None:
        self.audit_log.append(
            AuditEntry(
                at=ctx.now_utc or datetime.now(timezone.utc),
                tool_name=ctx.tool_name,
                verdict=decision.verdict,
                reason=decision.reason,
                policy_name=decision.policy_name,
            )
        )


# ---------------------------------------------------------------------------
# Starter policies
# ---------------------------------------------------------------------------


@dataclass
class AllowlistPolicy:
    """Generic single-tool allowlist on one input field.

    Example: AllowlistPolicy("send_email", field="to", allowed={"me@x.com"})
    denies email tool calls whose `to` is not in the allowed set.
    """

    tool_name: str
    field: str
    allowed: set[str]
    name: str = "allowlist"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.tool_name != self.tool_name:
            return PolicyDecision("allow", policy_name=self.name)
        value = ctx.tool_input.get(self.field)
        if value is None or str(value) in self.allowed:
            return PolicyDecision("allow", policy_name=self.name)
        return PolicyDecision(
            "deny",
            reason=f"'{self.field}={value}' not in allowlist for {self.tool_name}",
            policy_name=self.name,
        )


@dataclass
class CommandAllowlistPolicy:
    """argv[0]-basename allowlist for run_command (or any argv-shaped tool).

    Matches Path(argv[0]).name against `allowed`. Skipped for any other
    tool. Note: ambi/run_command.py's CommandPolicy already enforces this
    inside the tool — Warden adds this for defense-in-depth or if you
    want the deny decision visible in the Warden audit log.
    """

    allowed: set[str]
    tool_name: str = "run_command"
    name: str = "command_allowlist"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.tool_name != self.tool_name:
            return PolicyDecision("allow", policy_name=self.name)
        argv = ctx.tool_input.get("argv")
        if not isinstance(argv, list) or not argv:
            # Malformed input — let the tool's own validation reject it.
            return PolicyDecision("allow", policy_name=self.name)
        cmd = Path(str(argv[0])).name
        if cmd in self.allowed:
            return PolicyDecision("allow", policy_name=self.name)
        return PolicyDecision(
            "deny",
            reason=f"command '{cmd}' not in allowlist",
            policy_name=self.name,
        )


@dataclass
class ArgvValidatorPolicy:
    """Forbid specific argv patterns for run_command-shaped tools.

    Matches each forbidden pattern as a substring against the space-joined
    argv. Coarse but pragmatic; use this to lock out the truly dangerous
    invocations (e.g. `git push --force`, `rm -rf`).
    """

    forbid: list[str]
    tool_name: str = "run_command"
    name: str = "argv_validator"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.tool_name != self.tool_name:
            return PolicyDecision("allow", policy_name=self.name)
        argv = ctx.tool_input.get("argv")
        if not isinstance(argv, list):
            return PolicyDecision("allow", policy_name=self.name)
        joined = " ".join(str(a) for a in argv)
        for pattern in self.forbid:
            if pattern in joined:
                return PolicyDecision(
                    "deny",
                    reason=f"argv contains forbidden pattern: {pattern!r}",
                    policy_name=self.name,
                )
        return PolicyDecision("allow", policy_name=self.name)


@dataclass
class CostCeilingPolicy:
    """Deny tool calls once today's LLM spend has exceeded `daily_usd`.

    Coarse but effective: blocking tools stops the agent from compounding
    cost via further tool-driven turns. The model can still respond with
    text — it just can't act further until the budget rolls over.
    """

    usage_store: object  # UsageStore — typed loosely to avoid cycle
    daily_usd: float
    name: str = "cost_ceiling"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        now = ctx.now_utc or datetime.now(timezone.utc)
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            summary = await self.usage_store.summary(since=since)
        except Exception:
            return PolicyDecision("allow", policy_name=self.name)
        if summary.cost_usd >= self.daily_usd:
            return PolicyDecision(
                "deny",
                reason=(
                    f"daily spend ${summary.cost_usd:.4f} >= ceiling "
                    f"${self.daily_usd:.4f}"
                ),
                policy_name=self.name,
            )
        return PolicyDecision("allow", policy_name=self.name)


@dataclass
class QuietHoursPolicy:
    """Deny tool calls during a quiet-hours window in the local timezone.

    Useful for the scheduler — keep the agent from firing reminders at
    3am. Hours are integers 0–23. Window wraps midnight automatically
    when `start_hour > end_hour` (e.g. 22, 7 = 10pm–7am quiet).
    `tools=None` applies to every tool; pass a set to scope.
    """

    start_hour: int
    end_hour: int
    tools: set[str] | None = None
    name: str = "quiet_hours"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if self.tools is not None and ctx.tool_name not in self.tools:
            return PolicyDecision("allow", policy_name=self.name)
        now = ctx.now_utc or datetime.now(timezone.utc)
        local_hour = now.astimezone().hour
        if self.start_hour > self.end_hour:
            in_quiet = local_hour >= self.start_hour or local_hour < self.end_hour
        else:
            in_quiet = self.start_hour <= local_hour < self.end_hour
        if in_quiet:
            return PolicyDecision(
                "deny",
                reason=(
                    f"quiet hours {self.start_hour:02d}:00–"
                    f"{self.end_hour:02d}:00 (local hour {local_hour:02d})"
                ),
                policy_name=self.name,
            )
        return PolicyDecision("allow", policy_name=self.name)
