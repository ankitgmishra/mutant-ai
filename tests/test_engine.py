import pytest

from mutant.core.config import MutationConfig
from mutant.core.engine import MutationEngine, mutate, mutate_sync
from mutant.core.mutation import (
    BehaviorAnalysis,
    MutationCategory,
    MutationResult,
    MutationSeverity,
)
from mutant.core.registry import MutationRegistry
from mutant.core.scenario import Scenario
from mutant.dimensions.base import MutationDimension
from tests.conftest import MockLLMProvider


class _TestDimension(MutationDimension):
    id = "engine_test.dim"
    name = "Engine Test Dimension"
    description = "Used in engine tests."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return "Generate a test mutation."


@pytest.fixture
def engine_registry() -> MutationRegistry:
    reg = MutationRegistry()
    reg.register(_TestDimension())
    return reg


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(title="Engine Test", description="A scenario for engine testing.")


@pytest.fixture
def refund_scenario() -> Scenario:
    return Scenario(title="Refund", description="Customer refund request.")


@pytest.mark.asyncio
async def test_engine_run_returns_mutation_result(
    provider: MockLLMProvider, scenario: Scenario, engine_registry: MutationRegistry
) -> None:
    engine = MutationEngine(provider=provider, registry=engine_registry)
    result = await engine.run(
        scenario,
        config=MutationConfig(count=2, quality_review=False, deduplicate=False),
    )
    assert isinstance(result, MutationResult)
    assert isinstance(result.cases, list)
    assert isinstance(result.behavior_analysis, BehaviorAnalysis)


@pytest.mark.asyncio
async def test_engine_coverage_report_generated(
    provider: MockLLMProvider, scenario: Scenario, engine_registry: MutationRegistry
) -> None:
    engine = MutationEngine(provider=provider, registry=engine_registry)
    result = await engine.run(
        scenario, config=MutationConfig(count=1, deduplicate=False)
    )
    assert 0.0 <= result.coverage_score <= 1.0


@pytest.mark.asyncio
async def test_engine_empty_pool_raises(
    provider: MockLLMProvider, scenario: Scenario, engine_registry: MutationRegistry
) -> None:
    engine = MutationEngine(provider=provider, registry=engine_registry)
    with pytest.raises(ValueError, match="No dimensions match"):
        await engine.run(
            scenario, config=MutationConfig(count=1, dimension_ids=["nonexistent.dim"])
        )


@pytest.mark.asyncio
async def test_mutate_returns_result(
    provider: MockLLMProvider, scenario: Scenario, engine_registry: MutationRegistry
) -> None:
    result = await mutate(
        scenario,
        provider,
        registry=engine_registry,
        config=MutationConfig(
            count=1,
            analyze_coverage=False,
            quality_review=False,
            deduplicate=False,
            dimension_ids=["engine_test.dim"],
        ),
    )
    assert isinstance(result, MutationResult)


@pytest.mark.asyncio
async def test_mutate_uses_global_registry(
    provider: MockLLMProvider, refund_scenario: Scenario
) -> None:
    import mutant  # noqa: F401

    result = await mutate(
        refund_scenario,
        provider,
        config=MutationConfig(
            count=1,
            quality_review=False,
            deduplicate=False,
            dimension_ids=["emotion.angry"],
        ),
    )
    assert isinstance(result, MutationResult)


def test_mutate_sync_works(
    provider: MockLLMProvider, scenario: Scenario, engine_registry: MutationRegistry
) -> None:
    result = mutate_sync(
        scenario,
        provider,
        registry=engine_registry,
        config=MutationConfig(count=1, deduplicate=False),
    )
    assert isinstance(result, MutationResult)
