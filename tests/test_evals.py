"""Unit tests for the evals harness — parsing + assertion logic.

Live scenario runs require GEMINI_API_KEY and are exercised by
`uv run ambi eval`, not pytest. These tests are offline.
"""

import pytest

from ambi.evals import (
    Assertion,
    Scenario,
    ScenarioResult,
    check_assertion,
    load_scenarios,
)


# ---------- loading ----------


def test_load_simple_scenario(tmp_path):
    f = tmp_path / "one.yaml"
    f.write_text(
        "name: x\n"
        "description: y\n"
        "input: hello\n"
        "assert:\n"
        "  - text_contains: hi\n"
        "  - tool_not_called: recall_memory\n"
    )
    [scen] = load_scenarios(tmp_path)
    assert scen.name == "x"
    assert scen.description == "y"
    assert scen.input == "hello"
    assert len(scen.assertions) == 2
    assert scen.assertions[0].type == "text_contains"
    assert scen.assertions[0].value == "hi"


def test_load_name_defaults_to_filename(tmp_path):
    f = tmp_path / "thing.yaml"
    f.write_text("input: x\n")
    [scen] = load_scenarios(tmp_path)
    assert scen.name == "thing"


def test_load_directory_loads_all(tmp_path):
    (tmp_path / "a.yaml").write_text("input: a\n")
    (tmp_path / "b.yaml").write_text("input: b\n")
    scenarios = load_scenarios(tmp_path)
    assert {s.name for s in scenarios} == {"a", "b"}


def test_load_accepts_assertions_or_assert(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("input: x\nassertions:\n  - text_contains: foo\n")
    [scen] = load_scenarios(tmp_path)
    assert scen.assertions[0].value == "foo"


def test_load_rejects_missing_input(tmp_path):
    f = tmp_path / "broken.yaml"
    f.write_text("name: x\n")
    with pytest.raises(ValueError, match="missing required field 'input'"):
        load_scenarios(tmp_path)


def test_load_rejects_multi_key_assertion(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("input: x\nassert:\n  - text_contains: a\n    tool_called: b\n")
    with pytest.raises(ValueError, match="single-key mapping"):
        load_scenarios(tmp_path)


# ---------- assertion checks ----------


def _check(atype: str, value, **kw):
    return check_assertion(
        Assertion(type=atype, value=value),
        response_text=kw.get("response_text", ""),
        tools_called=kw.get("tools_called", []),
        input_tokens=kw.get("input_tokens", 0),
        output_tokens=kw.get("output_tokens", 0),
        cost_usd=kw.get("cost_usd", 0.0),
    )


def test_text_contains_case_insensitive_pass():
    r = _check("text_contains", "Churchill", response_text="Winston churchill was…")
    assert r.passed


def test_text_contains_fail():
    r = _check("text_contains", "Churchill", response_text="no relevant text")
    assert not r.passed
    assert "missing substring" in r.detail


def test_text_not_contains_pass():
    r = _check("text_not_contains", "happy to help", response_text="here you go.")
    assert r.passed


def test_text_not_contains_fail():
    r = _check("text_not_contains", "happy to help", response_text="I'm happy to help!")
    assert not r.passed


def test_text_matches_regex_pass():
    r = _check("text_matches", r"(?i)tokyo|jst", response_text="It's 7pm JST.")
    assert r.passed


def test_text_matches_regex_fail():
    r = _check("text_matches", r"(?i)tokyo|jst", response_text="No idea.")
    assert not r.passed


def test_text_not_matches_pass():
    r = _check("text_not_matches", r"(?i)i don't know", response_text="The answer is…")
    assert r.passed


def test_text_not_matches_fail():
    r = _check("text_not_matches", r"(?i)i don't know", response_text="I don't know.")
    assert not r.passed


def test_tool_called_pass():
    r = _check("tool_called", "get_current_time", tools_called=["get_current_time"])
    assert r.passed


def test_tool_called_fail():
    r = _check("tool_called", "recall_memory", tools_called=["get_current_time"])
    assert not r.passed


def test_tool_not_called_pass():
    r = _check("tool_not_called", "recall_memory", tools_called=["get_current_time"])
    assert r.passed


def test_tool_not_called_fail():
    r = _check("tool_not_called", "recall_memory", tools_called=["recall_memory", "x"])
    assert not r.passed


def test_max_input_tokens_pass():
    r = _check("max_input_tokens", 1000, input_tokens=500)
    assert r.passed


def test_max_input_tokens_fail():
    r = _check("max_input_tokens", 1000, input_tokens=1500)
    assert not r.passed


def test_max_cost_usd_pass():
    r = _check("max_cost_usd", 0.01, cost_usd=0.005)
    assert r.passed


def test_unknown_assertion_type_returns_failure():
    r = _check("not_a_real_type", "x")
    assert not r.passed
    assert "unknown assertion type" in r.detail


# ---------- ScenarioResult.passed property ----------


def test_scenario_result_passed_when_all_assertions_pass():
    from ambi.evals import AssertionResult

    r = ScenarioResult(
        scenario=Scenario(name="x", description="", input=""),
        response_text="", tools_called=[],
        input_tokens=0, output_tokens=0, cost_usd=0.0,
        error=None,
        assertion_results=[
            AssertionResult(Assertion("text_contains", "x"), True),
            AssertionResult(Assertion("tool_not_called", "y"), True),
        ],
    )
    assert r.passed


def test_scenario_result_fails_if_any_assertion_fails():
    from ambi.evals import AssertionResult

    r = ScenarioResult(
        scenario=Scenario(name="x", description="", input=""),
        response_text="", tools_called=[],
        input_tokens=0, output_tokens=0, cost_usd=0.0,
        error=None,
        assertion_results=[
            AssertionResult(Assertion("text_contains", "x"), True),
            AssertionResult(Assertion("tool_not_called", "y"), False, "fail"),
        ],
    )
    assert not r.passed


def test_scenario_result_fails_on_error_even_if_all_assertions_pass():
    r = ScenarioResult(
        scenario=Scenario(name="x", description="", input=""),
        response_text="", tools_called=[],
        input_tokens=0, output_tokens=0, cost_usd=0.0,
        error="boom",
        assertion_results=[],
    )
    assert not r.passed


# ---------- setup block ----------


def test_load_scenario_with_setup(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text(
        "input: hi\n"
        "setup:\n"
        "  env:\n"
        "    FOO: bar\n"
        "  prepare:\n"
        "    - create_obsidian_notes:\n"
        "        count: 10\n"
    )
    [scen] = load_scenarios(tmp_path)
    assert scen.setup["env"] == {"FOO": "bar"}
    assert scen.setup["prepare"][0] == {"create_obsidian_notes": {"count": 10}}


def test_load_rejects_non_dict_setup(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("input: hi\nsetup: not a dict\n")
    with pytest.raises(ValueError, match="setup' must be a mapping"):
        load_scenarios(tmp_path)


def test_apply_scenario_setup_env_override():
    import os
    from ambi.evals import Scenario, apply_scenario_setup

    saved = os.environ.get("TEST_EVAL_VAR")
    try:
        scen = Scenario(
            name="x", description="", input="",
            setup={"env": {"TEST_EVAL_VAR": "set-value"}},
        )
        with apply_scenario_setup(scen):
            assert os.environ["TEST_EVAL_VAR"] == "set-value"
        # Restored after exit.
        assert os.environ.get("TEST_EVAL_VAR") == saved
    finally:
        if saved is None:
            os.environ.pop("TEST_EVAL_VAR", None)
        else:
            os.environ["TEST_EVAL_VAR"] = saved


def test_apply_scenario_setup_tmp_dir_substitution():
    import os
    from ambi.evals import Scenario, apply_scenario_setup

    scen = Scenario(
        name="x", description="", input="",
        setup={"env": {"TEST_TMP_VAL": "{tmp_dir}/sub/path"}},
    )
    with apply_scenario_setup(scen) as tmp_dir:
        value = os.environ["TEST_TMP_VAL"]
        assert value.endswith("/sub/path")
        assert str(tmp_dir) in value
    os.environ.pop("TEST_TMP_VAL", None)


def test_apply_scenario_setup_create_obsidian_notes():
    from pathlib import Path
    from ambi.evals import Scenario, apply_scenario_setup

    scen = Scenario(
        name="x", description="", input="",
        setup={
            "env": {"X_VAULT": "{tmp_dir}/vault"},
            "prepare": [
                {"create_obsidian_notes": {
                    "count": 12,
                    "folders": ["Inbox", "Areas"],
                }},
            ],
        },
    )
    with apply_scenario_setup(scen) as tmp_dir:
        inbox = Path(tmp_dir) / "vault" / "Inbox"
        areas = Path(tmp_dir) / "vault" / "Areas"
        assert inbox.is_dir()
        assert areas.is_dir()
        assert len(list(inbox.glob("*.md"))) == 6
        assert len(list(areas.glob("*.md"))) == 6


def test_apply_scenario_setup_cleans_up_tmp_dir():
    from pathlib import Path
    from ambi.evals import Scenario, apply_scenario_setup

    scen = Scenario(
        name="x", description="", input="",
        setup={"prepare": [{"create_obsidian_notes": {"count": 3}}]},
    )
    with apply_scenario_setup(scen) as tmp_dir:
        path = Path(tmp_dir)
        assert path.exists()
    # Removed after exit.
    assert not path.exists()


def test_apply_scenario_setup_unknown_prepare_action_raises():
    from ambi.evals import Scenario, apply_scenario_setup

    scen = Scenario(
        name="x", description="", input="",
        setup={"prepare": [{"not_a_real_action": {}}]},
    )
    with pytest.raises(ValueError, match="unknown prepare action"):
        with apply_scenario_setup(scen):
            pass


def test_apply_scenario_setup_noop_for_empty_setup():
    from ambi.evals import Scenario, apply_scenario_setup

    scen = Scenario(name="x", description="", input="", setup={})
    with apply_scenario_setup(scen) as tmp_dir:
        assert tmp_dir is None


# ---------- usage capture (run_scenario) ----------


async def test_run_scenario_captures_usage_and_binds_assertions():
    """run_scenario attributes real tokens/cost from the TrackingProvider, so
    the max_*_tokens / max_cost_usd assertions actually enforce."""
    from ambi.evals import Assertion, Scenario, run_scenario
    from ambi.loop import Agent
    from ambi.tool import ToolRegistry
    from ambi.types import StreamEnd, TextChunk
    from ambi.usage import TrackingProvider, UsageStore

    from tests.mock_provider import MockStreamProvider

    inner = MockStreamProvider([
        [
            TextChunk(text="Tokyo is GMT+9."),
            StreamEnd(
                stop_reason="end_turn",
                usage={"input_tokens": 120, "output_tokens": 30,
                       "model": "gemini-2.5-flash"},
            ),
        ],
    ])
    provider = TrackingProvider(inner=inner, store=UsageStore(":memory:"))
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s")

    scen = Scenario(
        name="usage", description="", input="time in tokyo?",
        assertions=[
            Assertion("max_output_tokens", 10),   # 30 > 10 → must FAIL now
            Assertion("max_input_tokens", 1000),   # 120 <= 1000 → pass
        ],
    )
    result = await run_scenario(scen, agent)

    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.cost_usd > 0
    by_type = {r.assertion.type: r.passed for r in result.assertion_results}
    assert by_type["max_output_tokens"] is False   # actually enforced
    assert by_type["max_input_tokens"] is True


async def test_run_scenario_zero_usage_without_tracking_provider():
    """A bare provider (no usage_snapshot) → zeros, evals still run."""
    from ambi.evals import Scenario, run_scenario
    from ambi.loop import Agent
    from ambi.tool import ToolRegistry
    from ambi.types import StreamEnd, TextChunk

    from tests.mock_provider import MockStreamProvider

    provider = MockStreamProvider([
        [TextChunk(text="hi"), StreamEnd(stop_reason="end_turn")],
    ])
    agent = Agent(provider=provider, tools=ToolRegistry(), system="s")
    result = await run_scenario(
        Scenario(name="x", description="", input="hi", assertions=[]), agent
    )
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (0, 0, 0.0)


def test_register_prepare_action():
    from ambi.evals import (
        Scenario, apply_scenario_setup, register_prepare_action,
    )

    called = {}

    def custom_action(params, tmp_dir):
        called["params"] = params
        called["tmp_dir"] = tmp_dir

    register_prepare_action("test_custom_action", custom_action)
    scen = Scenario(
        name="x", description="", input="",
        setup={"prepare": [{"test_custom_action": {"foo": "bar"}}]},
    )
    with apply_scenario_setup(scen) as tmp_dir:
        assert called["params"] == {"foo": "bar"}
        assert called["tmp_dir"] == tmp_dir
