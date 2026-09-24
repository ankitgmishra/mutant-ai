"""Tests for pipeline stages in isolation."""

from __future__ import annotations

import pytest

from mutant.core.mutation import (
    MutationCase,
    MutationCategory,
    MutationResult,
    MutationSeverity,
)
from mutant.core.scenario import Scenario
from mutant.dimensions.base import MutationDimension
from mutant.pipeline.context import PipelineConfig, PipelineContext
from mutant.pipeline.stages import (
    _generate_dimension_batch,
    analyze_behavior,
    deduplicate,
    plan_mutations,
)
from mutant.providers.base import BaseLLMProvider, LLMResponse
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
    assert result.behavior_analysis.detected_domain != ""
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


# ── Probe-set coverage: count shortfall must be visible ──────────────────
#
# Measured in the example security notebooks: `mutate(count=8)` returned 1, 1, 2, 2 and
# 4 probes across dimensions. The batch call returns whatever the model produced, and
# the per-probe fallback only ran when it produced nothing — so a request for eight
# probes could quietly deliver one, and a thin probe set then looked like a clean run.


def _case(dimension_id: str, suffix: str) -> MutationCase:
    return MutationCase(
        id=f"case-{dimension_id}-{suffix}",
        dimension_id=dimension_id,
        dimension_name="Dimension",
        category=MutationCategory.SECURITY,
        severity=MutationSeverity.HIGH,
        original_description="original",
        mutated_description=f"probe {suffix}",
    )


def test_mutation_result_reports_the_shortfall_and_dimension_mix() -> None:
    result = MutationResult(
        cases=[
            _case("security.data_leakage", "a"),
            _case("security.data_leakage", "b"),
            _case("security.tool_argument_manipulation", "c"),
        ],
        requested_count=6,
    )
    assert result.shortfall == 3
    assert result.dimension_counts == {
        "security.data_leakage": 2,
        "security.tool_argument_manipulation": 1,
    }
    assert "WARNING" in result.summary
    assert "3 of 6" in result.summary


def test_mutation_result_without_a_requested_count_has_no_shortfall() -> None:
    result = MutationResult(cases=[_case("security.data_leakage", "a")])
    assert result.shortfall == 0
    assert "WARNING" not in result.summary


class _BatchOneThenSingles(BaseLLMProvider):
    """A small model: asked for N probes in one call, it returns one."""

    provider_name = "batch-attempt"

    def __init__(self, batch_items: int) -> None:
        self.model = "batch-attempt"
        self._batch_items = batch_items
        self.batch_calls = 0
        self.single_calls = 0

    async def complete(self, messages, *, temperature=0.8, max_tokens=4096):
        return LLMResponse(content="{}", model=self.model)

    async def complete_json(self, messages, schema, *, temperature=0.7, max_tokens=4096, max_retries=3):
        fields = set(getattr(schema, "model_fields", {}))
        if "plans" in fields:
            return schema.model_validate(
                {
                    "plans": [
                        {
                            "plan_id": f"p{i}",
                            "title": "t",
                            "behavioral_challenge": "b",
                            "transformation_description": "d",
                            "key_elements": [],
                            "avoid_elements": [],
                        }
                        for i in range(8)
                    ]
                }
            )
        if "mutations" in fields:
            self.batch_calls += 1
            return schema.model_validate(
                {
                    "mutations": [
                        {"mutated_description": f"batch probe {i}", "rationale": "r"}
                        for i in range(self._batch_items)
                    ]
                }
            )
        if "mutated_description" in fields:
            self.single_calls += 1
            return schema.model_validate(
                {"mutated_description": f"single probe {self.single_calls}", "rationale": "r"}
            )
        return schema.model_validate({})


@pytest.mark.asyncio
async def test_a_short_batch_is_topped_up_to_the_requested_count(ctx: PipelineContext) -> None:
    provider = _BatchOneThenSingles(batch_items=1)
    cases = await _generate_dimension_batch(ctx, provider, _MockDimension(), 3, [])
    assert len(cases) == 3
    assert provider.batch_calls == 1
    assert provider.single_calls == 2  # only the two that were missing


@pytest.mark.asyncio
async def test_a_full_batch_is_not_topped_up(ctx: PipelineContext) -> None:
    provider = _BatchOneThenSingles(batch_items=3)
    cases = await _generate_dimension_batch(ctx, provider, _MockDimension(), 3, [])
    assert len(cases) == 3
    assert provider.single_calls == 0


@pytest.mark.asyncio
async def test_an_empty_batch_falls_back_to_per_probe_generation(ctx: PipelineContext) -> None:
    provider = _BatchOneThenSingles(batch_items=0)
    cases = await _generate_dimension_batch(ctx, provider, _MockDimension(), 3, [])
    assert len(cases) == 3
    assert provider.single_calls == 3
