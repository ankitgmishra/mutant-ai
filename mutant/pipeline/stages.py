"""
mutant/pipeline/stages.py — V0.4
=================================
All pipeline stages. Each is a pure async function:
  PipelineContext × BaseLLMProvider → PipelineContext

New in V0.4:
  - quality_review stage (Stage 4)
  - Richer BehaviorAnalysis fields
  - MutationPlan (replaces BehaviorPlan) with why_selected, difficulty, diversity_strategy
  - EvaluationCase with realism_score, diversity_score, expected_failure_modes, sub_dimension
  - Updated coverage stage uses explored/unexplored dimensions
  - PipelineStats tracked throughout
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import anyio

from mutant.core.mutation import (
    BehaviorAnalysis,
    DeduplicationResult,
    EvaluationCase,
    MutationPlan,
    MutationPlanResponse,
    QualityReviewResult,
    QualityScore,
)
from mutant.exceptions import ProviderError
from mutant.pipeline.context import PipelineContext
from mutant.pipeline.prompts import render_prompt
from mutant.providers.base import BaseLLMProvider, LLMMessage, ParseError

if TYPE_CHECKING:
    from mutant.dimensions.base import MutationDimension


# ── Stage 1: Behavior Analysis ─────────────────────────────────────────────────


async def analyze_behavior(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
) -> PipelineContext:
    """Analyze the scenario's behavioral space. Sets ctx.behavior_analysis."""
    t0 = time.monotonic()
    prompt = render_prompt(
        "behavior_analysis.md",
        template_override=ctx.config.prompts.get("behavior_analysis"),
        scenario_title=ctx.scenario.title,
        scenario_description=ctx.scenario.description,
        scenario_tags=ctx.scenario.tags,
    )
    try:
        ctx.behavior_analysis = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            BehaviorAnalysis,
            temperature=0.3,
        )
    except Exception as e:
        raise ProviderError(
            "Failed to parse behavior analysis.",
            stage="behavior_analysis",
            provider_name=provider.provider_name,
            original_error=e,
        ) from e
    ctx.stage_timings["behavior_analysis"] = time.monotonic() - t0
    return ctx


# ── Stage 2: Mutation Planning ─────────────────────────────────────────────────


async def plan_mutations(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimensions: list[MutationDimension],
) -> PipelineContext:
    """Plan which dimensions to use and how many mutations each produces."""
    t0 = time.monotonic()
    dim_dicts = [
        {
            "id": d.id,
            "name": d.name,
            "category": d.category.value,
            "severity": d.severity.value,
            "description": d.description,
        }
        for d in dimensions
    ]
    prompt = render_prompt(
        "mutation_planning.md",
        template_override=ctx.config.prompts.get("mutation_planning"),
        scenario_title=ctx.scenario.title,
        scenario_description=ctx.scenario.description,
        behavior_analysis=ctx.behavior_analysis.model_dump()
        if ctx.behavior_analysis
        else {},
        dimensions=dim_dicts,
        target_count=ctx.config.count,
    )
    try:
        plan = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            MutationPlan,
            temperature=0.5,
        )
    except Exception as e:
        raise ProviderError(
            "Failed to generate mutation plan.",
            stage="mutation_planning",
            provider_name=provider.provider_name,
            original_error=e,
        ) from e
    # Normalize total_planned
    total = sum(a.count for a in plan.dimension_allocations)
    plan = plan.model_copy(update={"total_planned": total})
    ctx.mutation_plan = plan
    ctx.stage_timings["mutation_planning"] = time.monotonic() - t0
    return ctx


# ── Stage 3: Mutation Generation ───────────────────────────────────────────────


async def generate_mutations(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimensions_by_id: dict[str, MutationDimension],
) -> PipelineContext:
    """Generate all mutations concurrently, one batch per dimension."""
    t0 = time.monotonic()
    if ctx.mutation_plan is None:
        raise RuntimeError("mutation_plan must be set before generate_mutations.")

    all_cases: list[list[EvaluationCase]] = []

    async with anyio.create_task_group() as tg:
        for allocation in ctx.mutation_plan.dimension_allocations:
            dim = dimensions_by_id.get(allocation.dimension_id)
            if dim is None:
                continue

            async def _run(
                _dim: MutationDimension = dim,
                _alloc=allocation,
            ) -> None:
                cases = await _generate_dimension_batch(
                    ctx, provider, _dim, _alloc.count, _alloc.focus_areas
                )
                all_cases.append(cases)

            tg.start_soon(_run)

    flat: list[EvaluationCase] = [c for batch in all_cases for c in batch]
    ctx.raw_cases = flat
    ctx.stage_timings["mutation_generation"] = time.monotonic() - t0
    return ctx


async def _generate_dimension_batch(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimension: MutationDimension,
    count: int,
    focus_areas: list[str],
) -> list[EvaluationCase]:
    # Step A: Get specific mutation plans from LLM
    try:
        plan_prompt = render_prompt(
            "mutation_planning.md",
            template_override=ctx.config.prompts.get("mutation_planning"),
            scenario_title=ctx.scenario.title,
            scenario_description=ctx.scenario.description,
            behavior_analysis=ctx.behavior_analysis.model_dump()
            if ctx.behavior_analysis
            else {},
            dimensions=[
                {
                    "id": dimension.id,
                    "name": dimension.name,
                    "category": dimension.category.value,
                    "severity": dimension.severity.value,
                    "description": dimension.description,
                    "mutation_instructions": dimension.get_mutation_instructions(),
                }
            ],
            focus_areas=focus_areas,
            target_count=count,
        )
        plan_resp = await provider.complete_json(
            [LLMMessage(role="user", content=plan_prompt)],
            MutationPlanResponse,
            temperature=0.8,
        )
        plans = plan_resp.plans[:count]
    except (ParseError, Exception):
        plans = []

    # Step B: Generate mutations concurrently
    cases: list[EvaluationCase] = []

    # Dynamically build the response model ONCE outside the loop based on config toggles
    # (Pydantic create_model is heavy and shouldn't be called inside a concurrent worker loop)
    from pydantic import Field, create_model

    fields = {"mutated_description": (str, ...)}
    if ctx.config.generate_rationale:
        fields["rationale"] = (str, ...)
    if ctx.config.generate_tags:
        fields["behavioral_tags"] = (list[str], Field(default_factory=list))

    DynamicGeneratedMutation = create_model("DynamicGeneratedMutation", **fields)

    async def _gen_one(plan=None, idx: int = 0) -> None:
        try:
            gen_prompt = render_prompt(
                "mutation_generation.md",
                template_override=ctx.config.prompts.get("mutation_generation"),
                original_description=ctx.scenario.description,
                dimension_name=dimension.name,
                dimension_category=dimension.category.value,
                dimension_severity=dimension.severity.value,
                plan=plan.model_dump()
                if plan
                else {
                    "title": f"{dimension.name} variation {idx + 1}",
                    "behavioral_challenge": dimension.description,
                    "transformation_description": dimension.get_mutation_instructions(),
                    "key_elements": [],
                    "avoid_elements": [],
                },
                dimension_examples=dimension.get_examples()[:3],
                generate_rationale=ctx.config.generate_rationale,
                generate_tags=ctx.config.generate_tags,
            )
            generated = await provider.complete_json(
                [LLMMessage(role="user", content=gen_prompt)],
                DynamicGeneratedMutation,
                temperature=ctx.config.temperature,
            )
            case = EvaluationCase(
                id=str(uuid.uuid4()),
                dimension_id=dimension.id,
                dimension_name=dimension.name,
                category=dimension.category,
                severity=dimension.severity,
                original_description=ctx.scenario.description,
                mutated_description=generated.mutated_description,
                rationale=getattr(generated, "rationale", ""),
                behavioral_tags=getattr(generated, "behavioral_tags", []),
                realism_notes=getattr(generated, "realism_notes", ""),
                plan_title=plan.title if plan else "",
                generation_metadata={
                    "provider": provider.provider_name,
                    "temperature": ctx.config.temperature,
                },
            )
            cases.append(case)
        except Exception as e:
            import logging

            logging.getLogger("mutant").error(f"Error in _gen_one: {e}")
            pass

    async with anyio.create_task_group() as tg:
        for i, plan in enumerate(plans or [None] * count):  # type: ignore[list-item]
            tg.start_soon(_gen_one, plan, i)

    return cases


# ── Stage 4: Quality Review ────────────────────────────────────────────────────


async def quality_review(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
) -> PipelineContext:
    """Evaluate every mutation for realism, usefulness, diversity, and consistency.

    Batches cases into groups to avoid context overflow on smaller models.
    Updates realism_score and diversity_score on approved cases.
    Rejected cases are filtered out; their count is tracked in stats.
    """
    t0 = time.monotonic()

    if not ctx.raw_cases:
        ctx.reviewed_cases = []
        return ctx

    BATCH_SIZE = ctx.config.quality_batch_size
    all_scores: list[QualityScore] = []
    approved_ids: set[str] = set()
    rejected_ids: set[str] = set()

    # Chunk cases into batches for context-safety
    def _chunks(lst: list, n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    async def _review_batch(batch: list[EvaluationCase]) -> None:
        snapshots = [
            {
                "id": c.id,
                "dimension_name": c.dimension_name,
                "severity": c.severity.value,
                "mutated_description": c.mutated_description,
            }
            for c in batch
        ]
        try:
            prompt = render_prompt(
                "quality_review.md",
                template_override=ctx.config.prompts.get("quality_review"),
                original_description=ctx.scenario.description,
                cases=snapshots,
            )
            result = await provider.complete_json(
                [LLMMessage(role="user", content=prompt)],
                QualityReviewResult,
                temperature=0.3,
            )
            all_scores.extend(result.scores)
            approved_ids.update(result.approved_ids)
            rejected_ids.update(result.rejected_ids)
        except Exception:
            # On failure, approve entire batch (fail-open)
            for c in batch:
                approved_ids.add(c.id)

    async with anyio.create_task_group() as tg:
        for batch in _chunks(ctx.raw_cases, BATCH_SIZE):
            tg.start_soon(_review_batch, batch)

    # Build score lookup
    score_by_id = {s.case_id: s for s in all_scores}

    # Update cases with review scores; keep only approved
    reviewed: list[EvaluationCase] = []
    for case in ctx.raw_cases:
        score = score_by_id.get(case.id)
        if score and not score.approved:
            rejected_ids.add(case.id)
            continue
        # Enrich case with quality scores
        updates: dict = {}
        if score:
            updates["quality_approved"] = score.approved
        reviewed.append(case.model_copy(update=updates) if updates else case)

    ctx.reviewed_cases = reviewed
    ctx.quality_review_result = QualityReviewResult(
        scores=all_scores,
        approved_ids=list(approved_ids),
        rejected_ids=list(rejected_ids),
    )
    ctx.stage_timings["quality_review"] = time.monotonic() - t0
    return ctx


# ── Stage 5: Deduplication ────────────────────────────────────────────────────


async def deduplicate(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
) -> PipelineContext:
    """Semantic deduplication over reviewed cases."""
    t0 = time.monotonic()
    source = ctx.reviewed_cases or ctx.raw_cases

    if len(source) <= 1:
        ctx.deduplicated_cases = source
        return ctx

    snapshots = [
        {
            "id": c.id,
            "dimension_name": c.dimension_name,
            "mutated_description": c.mutated_description,
        }
        for c in source
    ]
    prompt = render_prompt(
        "deduplication.md",
        template_override=ctx.config.prompts.get("deduplication"),
        original_description=ctx.scenario.description,
        cases=snapshots,
    )
    try:
        result = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            DeduplicationResult,
            temperature=0.2,
        )
        keep_ids: set[str] = set(result.unique_ids)
        for group in result.duplicate_groups:
            keep_ids.add(group["primary_id"])

        dup_ids = {
            did
            for group in result.duplicate_groups
            for did in group.get("duplicate_ids", [])
        }

        deduped: list[EvaluationCase] = []
        for case in source:
            if case.id in dup_ids and case.id not in keep_ids:
                continue
            deduped.append(case)

        ctx.deduplicated_cases = deduped
    except Exception:
        ctx.deduplicated_cases = source
    ctx.stage_timings["deduplication"] = time.monotonic() - t0
    return ctx
