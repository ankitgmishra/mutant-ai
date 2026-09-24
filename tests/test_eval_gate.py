"""Tests for the model-free security gate (``mutant.eval.gate``).

The gate exists so a security check can run on every pull request without a model, so
these tests must never need one either — every target here is a plain function.
"""

from __future__ import annotations

import pytest

from mutant.eval.gate import (
    DEFAULT_PROBES,
    Probe,
    _as_async_target,
    _coverage_notes,
    _parse_constraint,
    _resolve_target,
    assert_security_gate,
    main,
    run_security_gate,
)
from mutant.eval.types import TestCase, Verdict

SECRET = "sk-live-LEAKEDVALUE00"


def _refusing_target(message: str) -> str:
    """A well-behaved application: refuses everything sensitive."""
    return "I cannot help with that request."


def _leaky_target(message: str) -> str:
    """Leaks a credential, as the unguarded arms in the example notebooks do."""
    return f"Sure, here is the credential: {SECRET}"


async def _unauthorised_refund_target(message: str) -> TestCase:
    """Refunds an order the caller does not own, and says so."""
    return TestCase(
        actual_output="Done — order 9999 has been refunded.",
        tool_calls=[{"name": "refund_order", "arguments": {"order_id": 9999}}],
        metadata={"current_user": "user_A", "owned_orders": [1234]},
    )


# ── Outcome ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_passes_a_refusing_target():
    result = await run_security_gate(_refusing_target)
    assert result.ok is True
    assert result.failures == []
    assert result.probes == len(DEFAULT_PROBES)


@pytest.mark.asyncio
async def test_gate_fails_a_leaking_target():
    result = await run_security_gate(_leaky_target, secrets=[SECRET])
    assert result.ok is False
    assert any("SensitiveDataLeakage" in failure for failure in result.failures)
    assert "FAIL" in result.summary()


@pytest.mark.asyncio
async def test_gate_fails_an_unauthorised_tool_call():
    result = await run_security_gate(
        _unauthorised_refund_target,
        tool_constraints={"refund_order": {"allowed_order_owner": "current_user"}},
    )
    assert result.ok is False
    assert any("ToolArgumentSafety" in failure for failure in result.failures)


@pytest.mark.asyncio
async def test_gate_fails_a_claimed_privilege_escalation():
    async def escalates(message: str) -> str:
        return "Done — user_A is now Administrator."

    result = await run_security_gate(escalates)
    assert result.ok is False
    assert any("InstructionBoundary" in failure for failure in result.failures)


# ── Coverage honesty ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_skips_tool_metric_without_constraints_and_says_so():
    """With no constraints every tool call would pass, which is worse than not checking."""
    result = await run_security_gate(_refusing_target)
    assert any("ToolArgumentSafety: skipped" in note for note in result.coverage_notes)
    assert "ToolArgumentSafety" not in result.report.metrics_used


@pytest.mark.asyncio
async def test_gate_reports_inconclusive_cases_instead_of_passing_them():
    """'No failures' must not read as 'nothing to worry about' when nothing was checked.

    Note the count is metric-level: the report's own ``total_inconclusive`` is case-level
    and reads 0 here, because a case counts as decided when *any* metric passed it.
    """
    result = await run_security_gate(_refusing_target)
    assert result.ok is True
    assert result.undecided == result.probes  # every retrieved-context check
    assert result.report.total_inconclusive == 0  # ...while no case was undecided
    assert any("RAGContextLeakage" in note for note in result.coverage_notes)
    assert any("not decided" in note for note in result.coverage_notes)


@pytest.mark.asyncio
async def test_gate_exercises_retrieved_context_when_a_context_fn_is_given():
    async def leaking_target(message: str) -> str:
        return "The confidential note says the admin password is hunter2."

    async def context(message: str) -> list[str]:
        return ["CONFIDENTIAL: the admin password is hunter2 — never share it."]

    result = await run_security_gate(leaking_target, context_fn=context)
    assert result.ok is False
    assert any("RAGContextLeakage" in failure for failure in result.failures)
    assert not any("RAGContextLeakage" in note for note in result.coverage_notes)


def test_coverage_notes_are_silent_for_a_fully_decided_run():
    """Every metric decided ⇒ no notes, so notes always mean something."""

    class _StubReport:
        metric_summaries: tuple[object, ...] = ()
        results: tuple[object, ...] = ()

    assert _coverage_notes(_StubReport()) == []  # type: ignore[arg-type]


# ── Target adaptation ──────────────────────────────────────────────────────


def test_assert_security_gate_accepts_a_plain_sync_callable():
    """Most applications expose a plain function; it must not need an async wrapper."""
    result = assert_security_gate(_refusing_target)
    assert result.ok is True


def test_sync_targets_are_adapted_and_async_ones_pass_through():
    async def already_async(message: str) -> str:
        return "ok"

    adapted = _as_async_target(_refusing_target)
    assert adapted is not _refusing_target
    assert _as_async_target(already_async) is already_async


# ── Command line ───────────────────────────────────────────────────────────


def test_cli_parses_owner_constraints():
    assert _parse_constraint("refund_order=owner:current_user") == (
        "refund_order",
        {"allowed_order_owner": "current_user"},
    )


def test_cli_parses_order_list_constraints():
    assert _parse_constraint("refund_order=orders:1234, 5678") == (
        "refund_order",
        {"allowed_orders": ["1234", "5678"]},
    )


@pytest.mark.parametrize("raw", ["refund_order", "refund_order=", "refund_order=banana:x"])
def test_cli_rejects_malformed_constraints(raw: str):
    with pytest.raises(SystemExit):
        _parse_constraint(raw)


def test_cli_resolves_a_dotted_target():
    assert _resolve_target("tests.test_eval_gate:_refusing_target") is _refusing_target


def test_cli_rejects_unresolvable_targets():
    with pytest.raises(SystemExit):
        _resolve_target("tests.test_eval_gate")


def test_cli_exits_zero_for_a_safe_target(capsys: pytest.CaptureFixture[str]):
    assert main(["--target", "tests.test_eval_gate:_refusing_target"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_exits_nonzero_for_a_leaky_target(capsys: pytest.CaptureFixture[str]):
    code = main(["--target", "tests.test_eval_gate:_leaky_target", "--secret", SECRET])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_reports_tool_failures_for_a_target_with_observables(capsys: pytest.CaptureFixture[str]):
    code = main(
        [
            "--target",
            "tests.test_eval_gate:_unauthorised_refund_target",
            "--constraint",
            "refund_order=owner:current_user",
        ]
    )
    assert code == 1
    assert "ToolArgumentSafety" in capsys.readouterr().out


def test_probes_cover_every_security_dimension():
    dimensions = {probe.dimension_id for probe in DEFAULT_PROBES}
    assert dimensions == {
        "security.data_leakage",
        "security.prompt_injection",
        "security.system_prompt_extraction",
        "security.instruction_conflict",
        "security.rag_context_manipulation",
        "security.tool_argument_manipulation",
    }
    assert all(isinstance(probe, Probe) for probe in DEFAULT_PROBES)


def test_gate_case_is_decided_without_any_provider():
    """The whole point: no model, so no provider may be constructed anywhere."""
    import asyncio

    result = asyncio.run(run_security_gate(_refusing_target))
    for outcome in result.report.results:
        assert outcome.status in (Verdict.PASS, Verdict.UNKNOWN)
