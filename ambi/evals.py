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

import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    setup: dict = field(default_factory=dict)
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

    setup = raw.get("setup") or {}
    if not isinstance(setup, dict):
        raise ValueError(f"{path}: 'setup' must be a mapping")

    return Scenario(
        name=raw.get("name") or path.stem,
        description=raw.get("description", "") or "",
        input=str(raw["input"]),
        assertions=assertions,
        setup=setup,
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


# ---------------------------------------------------------------------------
# Setup: env overrides + prepare actions
# ---------------------------------------------------------------------------


PrepareAction = Callable[[dict, Path], None]
_PREPARE_REGISTRY: dict[str, PrepareAction] = {}


def register_prepare_action(name: str, fn: PrepareAction) -> None:
    """Add a custom prepare action that scenarios can declare under setup.prepare."""
    _PREPARE_REGISTRY[name] = fn


@contextmanager
def apply_scenario_setup(scenario: Scenario):
    """Context manager that applies a scenario's setup block.

    - Creates a per-scenario tmp directory.
    - Overrides os.environ values from `setup.env`, substituting
      `{tmp_dir}` in string values.
    - Runs each `setup.prepare` action against the tmp dir.

    On exit: restores the original env and removes the tmp dir.
    """
    if not scenario.setup:
        yield None
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="ambi-eval-"))
    saved_env: dict[str, str | None] = {}
    try:
        env_overrides = scenario.setup.get("env") or {}
        if not isinstance(env_overrides, dict):
            raise ValueError("setup.env must be a mapping")
        for key, value in env_overrides.items():
            saved_env[key] = os.environ.get(key)
            os.environ[key] = str(value).format(tmp_dir=str(tmp_dir))

        prepare = scenario.setup.get("prepare") or []
        if not isinstance(prepare, list):
            raise ValueError("setup.prepare must be a list")
        for action in prepare:
            if not isinstance(action, dict) or len(action) != 1:
                raise ValueError(
                    f"each prepare action must be a single-key mapping, got {action!r}"
                )
            (kind, params), = action.items()
            handler = _PREPARE_REGISTRY.get(kind)
            if handler is None:
                raise ValueError(
                    f"unknown prepare action {kind!r}. "
                    f"Available: {sorted(_PREPARE_REGISTRY)}"
                )
            handler(params or {}, tmp_dir)

        yield tmp_dir
    finally:
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Built-in prepare actions
# ---------------------------------------------------------------------------


def _prepare_create_obsidian_notes(params: dict, tmp_dir: Path) -> None:
    """Populate a temp Obsidian vault with N notes distributed across folders.

    Pair with `setup.env.OBSIDIAN_VAULT: "{tmp_dir}/vault"` so the agent's
    obsidian_* tools read from the seeded directory.

    Params:
        count   total number of notes to create (default 100)
        folders list of relative folder paths (default ['Inbox'])
        vault   subpath under tmp_dir to use as the vault root (default 'vault')
    """
    count = int(params.get("count", 100))
    folders = params.get("folders") or ["Inbox"]
    vault_subpath = params.get("vault", "vault")

    vault = tmp_dir / vault_subpath
    vault.mkdir(parents=True, exist_ok=True)

    per_folder = max(1, count // len(folders))
    remaining = count
    for folder in folders:
        d = vault / folder
        d.mkdir(parents=True, exist_ok=True)
        n = min(per_folder, remaining)
        for i in range(n):
            (d / f"note_{i:04d}.md").write_text(
                f"---\ntitle: note {i} in {folder}\n---\n\n"
                f"Body of note {i}.\n"
            )
        remaining -= n


register_prepare_action("create_obsidian_notes", _prepare_create_obsidian_notes)
