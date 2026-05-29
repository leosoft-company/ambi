import pytest

from ambi.sensegate import (
    SenseGate,
    ToolInvocation,
    Verdict,
    _parse_verdict,
)
from ambi.types import ToolResultBlock, ToolUseBlock


class ScriptedVerifier:
    """Returns verdicts in order; records every call for inspection."""

    def __init__(self, verdicts: list[Verdict]):
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def verify(self, final_text, invocations):
        self.calls.append({"final_text": final_text, "invocations": list(invocations)})
        if not self.verdicts:
            raise RuntimeError("ScriptedVerifier out of verdicts")
        return self.verdicts.pop(0)


def _inv(name: str, kind: str = "read", error: bool = False) -> ToolInvocation:
    return ToolInvocation(
        call=ToolUseBlock(id=f"id-{name}", name=name, input={}),
        result=ToolResultBlock(
            tool_use_id=f"id-{name}",
            content="error" if error else "ok",
            is_error=error,
            _tool_name=name,
        ),
        kind=kind,
    )


# ---------- SenseGate behaviour ----------


async def test_no_invocations_skips_verifier():
    verifier = ScriptedVerifier([])  # would raise if called
    gate = SenseGate(verifier)
    verdict = await gate.check("hi there", [])
    assert verdict.matches is True
    assert verifier.calls == []
    assert gate.audit_log == []


async def test_match_does_not_log():
    verifier = ScriptedVerifier([Verdict(matches=True, reason="all consistent")])
    gate = SenseGate(verifier)
    verdict = await gate.check("I read the file.", [_inv("read_file", "read")])
    assert verdict.matches is True
    assert gate.audit_log == []


async def test_mismatch_logs_audit_entry():
    verifier = ScriptedVerifier(
        [Verdict(matches=False, reason="claimed send_email but no email tool called")]
    )
    gate = SenseGate(verifier)
    invocations = [_inv("read_file", "read")]
    verdict = await gate.check("I sent the email.", invocations)
    assert verdict.matches is False
    assert len(gate.audit_log) == 1
    entry = gate.audit_log[0]
    assert entry.had_write is False
    assert "send_email" in entry.reason
    assert entry.final_text_excerpt == "I sent the email."
    assert entry.invocations == invocations


async def test_verify_reads_false_skips_on_read_only_turn():
    """No LLM call, no audit entry when the turn has only read tools."""
    verifier = ScriptedVerifier([])  # would raise if called
    gate = SenseGate(verifier, verify_reads=False)
    verdict = await gate.check(
        "Recalled your preferences.", [_inv("recall_memory", "read")],
    )
    assert verdict.matches is True
    assert "skipped" in verdict.reason
    assert verifier.calls == []
    assert gate.audit_log == []


async def test_verify_reads_false_still_verifies_when_any_write_present():
    """A mixed turn (read + write) still goes through the verifier."""
    verifier = ScriptedVerifier(
        [Verdict(matches=False, reason="claimed write succeeded but didn't")]
    )
    gate = SenseGate(verifier, verify_reads=False)
    verdict = await gate.check(
        "Recalled and saved.",
        [_inv("recall_memory", "read"), _inv("update_memory", "write")],
    )
    assert verdict.matches is False
    assert len(verifier.calls) == 1
    assert len(gate.audit_log) == 1


async def test_verify_reads_true_default_still_verifies_reads():
    """Default behaviour unchanged — read-only turn still goes to verifier."""
    verifier = ScriptedVerifier([Verdict(matches=True, reason="ok")])
    gate = SenseGate(verifier)  # default verify_reads=True
    await gate.check("read ok", [_inv("recall_memory", "read")])
    assert len(verifier.calls) == 1


async def test_audit_entry_marks_writes():
    verifier = ScriptedVerifier([Verdict(matches=False, reason="wrong amount")])
    gate = SenseGate(verifier)
    await gate.check(
        "Transferred $100.",
        [_inv("read_balance", "read"), _inv("send_money", "write")],
    )
    assert gate.audit_log[0].had_write is True


# ---------- _parse_verdict ----------


def test_parse_plain_json():
    v = _parse_verdict('{"matches": true, "reason": "fine"}')
    assert v.matches is True
    assert v.reason == "fine"


def test_parse_strips_markdown_fences():
    v = _parse_verdict('```json\n{"matches": false, "reason": "no receipt"}\n```')
    assert v.matches is False
    assert v.reason == "no receipt"


def test_parse_strips_bare_fences():
    v = _parse_verdict('```\n{"matches": true, "reason": "ok"}\n```')
    assert v.matches is True


def test_parse_unparseable_fails_closed():
    v = _parse_verdict("this is not json at all")
    assert v.matches is False
    assert "no JSON" in v.reason or "unparseable" in v.reason


def test_parse_missing_key_fails_closed():
    v = _parse_verdict('{"verdict": "yes"}')
    assert v.matches is False


def test_parse_tolerates_leading_prose():
    v = _parse_verdict(
        'Sure, here is the verdict: {"matches": true, "reason": "fine"}'
    )
    assert v.matches is True
    assert v.reason == "fine"


def test_parse_tolerates_trailing_prose():
    v = _parse_verdict(
        '{"matches": false, "reason": "no receipt"} (some trailing notes)'
    )
    assert v.matches is False


def test_parse_fenced_with_language_tag_and_no_trailing_newline():
    # Common Gemini Flash shape: opening fence has language tag, closing fence
    # may have trailing whitespace or be followed by other content.
    v = _parse_verdict('```json\n{"matches": true, "reason": "ok"}\n```\n')
    assert v.matches is True
