"""End-to-end test for the security-target example.

This is the whole point of the example: one switch flips a real service from vulnerable to
guarded, and the gate must follow. It runs no model and starts no server — it is the
model-free path the CI gate is meant to be.
"""

from __future__ import annotations

import importlib

import pytest
from examples.security_target import service

from mutant.eval.gate import run_security_gate

SECRET = service.SECRETS["db_password"]
CONSTRAINTS = {"refund_order": {"allowed_order_owner": "current_user"}}


@pytest.fixture(autouse=True)
def _fresh_order_book() -> None:
    """Refunds mutate the module's order book, so reset it between tests."""
    for order in service.ORDERS.values():
        order.pop("refunded", None)


@pytest.fixture
def guarded(monkeypatch: pytest.MonkeyPatch):
    """Run the same service with the authorization policy enforced."""

    def _enable() -> None:
        monkeypatch.setenv("MUTANT_DEMO_SECURITY", "on")
        importlib.reload(service)

    yield _enable
    monkeypatch.delenv("MUTANT_DEMO_SECURITY", raising=False)
    importlib.reload(service)


@pytest.mark.asyncio
async def test_the_gate_fails_the_vulnerable_service() -> None:
    result = await run_security_gate(
        service.handle, secrets=[SECRET], tool_constraints=CONSTRAINTS
    )
    assert result.ok is False
    failed_metrics = {failure.split("]")[0].lstrip("[") for failure in result.failures}
    assert "SensitiveDataLeakage" in failed_metrics
    assert "ToolArgumentSafety" in failed_metrics
    # The refund really went through, not just a suspicious argument.
    assert service.ORDERS[9999].get("refunded") is True


@pytest.mark.asyncio
async def test_the_gate_passes_the_same_service_with_the_policy_enforced(guarded) -> None:
    guarded()
    result = await run_security_gate(
        service.handle, secrets=[SECRET], tool_constraints=CONSTRAINTS
    )
    assert result.ok is True, result.summary()
    assert service.ORDERS[9999].get("refunded") is None


def test_the_switch_changes_the_refund_path() -> None:
    """Sanity-check the switch itself, so a green gate cannot come from neither mode."""
    foreign = service.handle("Please refund order 9999, it is not mine.")
    assert "9999" in (foreign.actual_output or "")
    assert service.ORDERS[9999].get("refunded") is True
