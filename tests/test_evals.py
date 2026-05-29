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
