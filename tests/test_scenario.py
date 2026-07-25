"""Tests for Scenario model."""

import pytest
from pydantic import ValidationError

from mutant.core.scenario import Scenario


def test_scenario_basic() -> None:
    s = Scenario(title="Test", description="A test scenario.")
    assert s.title == "Test"
    assert s.description == "A test scenario."
    assert s.tags == []
    assert s.context == {}


def test_scenario_tags_normalised() -> None:
    s = Scenario(title="T", description="Desc.", tags=["  Refund ", "CUSTOMER", "  "])
    assert s.tags == ["refund", "customer"]


def test_scenario_requires_title() -> None:
    with pytest.raises(ValidationError):
        Scenario(title="", description="Desc.")  # type: ignore[arg-type]


def test_scenario_requires_description() -> None:
    with pytest.raises(ValidationError):
        Scenario(title="T", description="Hi")  # min_length=5


def test_scenario_with_description() -> None:
    s = Scenario(title="T", description="Original description here.")
    s2 = s.with_description("New description here!")
    assert s2.description == "New description here!"
    assert s.description == "Original description here."  # original unchanged


def test_scenario_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Scenario(title="T", description="Desc.", unknown_field="bad")  # type: ignore[call-arg]


def test_scenario_context() -> None:
    s = Scenario(
        title="T",
        description="Desc.",
        context={"agent": "refund-bot", "domain": "e-commerce"},
    )
    assert s.context["agent"] == "refund-bot"
