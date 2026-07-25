"""Tests for pipeline stages in isolation."""

from __future__ import annotations

import pytest

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.core.scenario import Scenario
from mutant.dimensions.base import MutationDimension
from mutant.pipeline.context import PipelineConfig, PipelineContext
from mutant.pipeline.stages import analyze_behavior, deduplicate, plan_mutations
from tests.conftest import MockLLMProvider


class _MockDimension(MutationDimension):
    id = "pipeline_test.dim"
    name = "Pipeline Test"
    description = "Pipeline test dimension."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return "Test instruction."


@pytest.fixture
def ctx() -> PipelineContext:
    scenario = Scenario(
        title="Test", description="A test scenario for pipeline testing."
    )
    config = PipelineConfig(count=3)
    return PipelineContext(scenario=scenario, config=config)


@pytest.mark.asyncio
async def test_analyze_behavior_populates_context(ctx: PipelineContext) -> None:
    provider = MockLLMProvider()
    result = await analyze_behavior(ctx, provider)
    assert result.behavior_analysis is not None
    assert result.behavior_analysis.intent != ""
    assert "behavior_analysis" in result.stage_timings


@pytest.mark.asyncio
async def test_plan_mutations_populates_context(ctx: PipelineContext) -> None:
    provider = MockLLMProvider()
    ctx = await analyze_behavior(ctx, provider)
    dims = [_MockDimension()]
    result = await plan_mutations(ctx, provider, dims)
    assert result.mutation_plan is not None
    assert len(result.mutation_plan.dimension_allocations) >= 1
    assert "mutation_planning" in result.stage_timings


@pytest.mark.asyncio
async def test_deduplicate_preserves_cases_when_no_dups(ctx: PipelineContext) -> None:
    import uuid

    from mutant.core.mutation import MutationCase

    provider = MockLLMProvider()

    # Pre-populate raw_cases
    case = MutationCase(
        id=str(uuid.uuid4()),
        dimension_id="test.dim",
        dimension_name="Test",
        category=MutationCategory.CONTEXT,
        severity=MutationSeverity.LOW,
        original_description="original",
        mutated_description="mutated version here",
        rationale="test rationale",
    )
    ctx.raw_cases = [case]

    result = await deduplicate(ctx, provider)
    # All cases should be present (mock returns no duplicates)
    assert len(result.deduplicated_cases) >= 1


@pytest.mark.asyncio
async def test_deduplicate_handles_empty_cases(ctx: PipelineContext) -> None:
    provider = MockLLMProvider()
    ctx.raw_cases = []
    result = await deduplicate(ctx, provider)
    assert result.deduplicated_cases == []


def test_pipeline_context_output_cases_fallback() -> None:
    scenario = Scenario(title="T", description="Description here.")
    config = PipelineConfig(count=1)
    ctx = PipelineContext(scenario=scenario, config=config)
    assert ctx.output_cases == []
