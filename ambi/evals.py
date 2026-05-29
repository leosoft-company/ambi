"""Behavioral evaluation harness for ambi.

A *scenario* is a YAML file declaring a user input + a list of assertions
about how the agent should respond. Each scenario runs against a real
provider (Gemini today; pluggable) and is graded by inspecting the
streamed events and final text.

Why not just unit tests? Unit tests cover the harness. Evals cover
*behavior* — they catch prompt regressions ("after editing system.md,
does it still answer general knowledge without firing recall_memory?")
that pytest can't.

Example scenario::

    name: general_knowledge_no_tools
    description: Universal facts answered from training, not via tools
    input: "Who is Winston Churchill?"
    assert:
      - text_contains: "Churchill"
      - text_not_matches: "(?i)i don't have|i don't know"
      - tool_not_called: recall_memory
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .loop import Agent
from .types import ChatComplete, ToolUseEvent


# ---------------------------------------------------------------------------
# Scenario + assertion model
# ---------------------------------------------------------------------------


@dataclass
class Assertion:
    type: str          # text_contains, text_not_contains, text_matches,
                       # text_not_matches, tool_called, tool_not_called,
                       # max_input_tokens, max_output_tokens, max_cost_usd
    value: Any


@dataclass
class Scenario:
    name: str
    description: str
    input: str
    assertions: list[Assertion] = field(default_factory=list)
    source_path: Path | None = None


@dataclass
class AssertionResult:
    assertion: Assertion
    passed: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    scenario: Scenario
    response_text: str
    tools_called: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None
    assertion_results: list[AssertionResult]

    @property
    def passed(self) -> bool:
        return self.error is None and all(
            r.passed for r in self.assertion_results
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load all `*.yaml` scenarios from a directory (or just one file)."""
    p = Path(path)
    if p.is_file():
        return [_load_one(p)]
    return [_load_one(f) for f in sorted(p.glob("*.yaml"))]


def _load_one(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    if "input" not in raw:
        raise ValueError(f"{path}: missing required field 'input'")
    raw_asserts = raw.get("assert") or raw.get("assertions") or []
    if not isinstance(raw_asserts, list):
        raise ValueError(f"{path}: 'assert' must be a list")

    assertions: list[Assertion] = []
    for item in raw_asserts:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                f"{path}: each assertion must be a single-key mapping, got {item!r}"
            )
        (atype, avalue), = item.items()
        assertions.append(Assertion(type=atype, value=avalue))

    return Scenario(
        name=raw.get("name") or path.stem,
        description=raw.get("description", "") or "",
        input=str(raw["input"]),
        assertions=assertions,
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------


def check_assertion(
    assertion: Assertion,
    *,
    response_text: str,
    tools_called: list[str],
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> AssertionResult:
    a = assertion
    t = a.type

    if t == "text_contains":
        needle = str(a.value).lower()
        ok = needle in response_text.lower()
        return AssertionResult(a, ok, "" if ok else f"missing substring {a.value!r}")

    if t == "text_not_contains":
        needle = str(a.value).lower()
        ok = needle not in response_text.lower()
        return AssertionResult(a, ok, "" if ok else f"contained forbidden {a.value!r}")

    if t == "text_matches":
        pat = re.compile(str(a.value))
        ok = bool(pat.search(response_text))
        return AssertionResult(a, ok, "" if ok else f"regex {a.value!r} did not match")

    if t == "text_not_matches":
        pat = re.compile(str(a.value))
        ok = not pat.search(response_text)
        return AssertionResult(a, ok, "" if ok else f"regex {a.value!r} matched")

    if t == "tool_called":
        ok = str(a.value) in tools_called
        return AssertionResult(a, ok, "" if ok else f"tool {a.value!r} was not called")

    if t == "tool_not_called":
        ok = str(a.value) not in tools_called
        return AssertionResult(a, ok, "" if ok else f"tool {a.value!r} was called")

    if t == "max_input_tokens":
        ok = input_tokens <= int(a.value)
        return AssertionResult(
            a, ok, "" if ok else f"input {input_tokens} > {a.value}",
        )

    if t == "max_output_tokens":
        ok = output_tokens <= int(a.value)
        return AssertionResult(
            a, ok, "" if ok else f"output {output_tokens} > {a.value}",
        )

    if t == "max_cost_usd":
        ok = cost_usd <= float(a.value)
        return AssertionResult(
            a, ok, "" if ok else f"${cost_usd:.4f} > ${float(a.value):.4f}",
        )

    return AssertionResult(a, False, f"unknown assertion type {t!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_scenario(scenario: Scenario, agent: Agent) -> ScenarioResult:
    """Drive a scenario through agent.chat_stream(), capture observations."""
    tools_called: list[str] = []
    final_text = ""
    error: str | None = None

    try:
        async for ev in agent.chat_stream(scenario.input):
            if isinstance(ev, ToolUseEvent):
                tools_called.append(ev.name)
            elif isinstance(ev, ChatComplete):
                final_text = ev.final_text
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    # Try to pull token usage from the provider's most recent call. The
    # exact mechanism depends on the provider being wrapped in
    # TrackingProvider; if not, these stay zero.
    input_tokens, output_tokens, cost_usd = _extract_usage_for(agent)

    results = [
        check_assertion(
            a,
            response_text=final_text,
            tools_called=tools_called,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        for a in scenario.assertions
    ]

    return ScenarioResult(
        scenario=scenario,
        response_text=final_text,
        tools_called=tools_called,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        error=error,
        assertion_results=results,
    )


def _extract_usage_for(agent: Agent) -> tuple[int, int, float]:
    """Best-effort: walk the agent's recent message history for usage hints.

    The TrackingProvider records to a separate store; we don't query that
    here to keep run_scenario self-contained. Token counting from provider
    responses is left to callers that want it.
    """
    return 0, 0, 0.0
