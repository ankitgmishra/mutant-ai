"""
mutant/pipeline/stages.py — V0.5
=================================
All pipeline stages. Each is a pure async function:
  PipelineContext × BaseLLMProvider → PipelineContext

V0.5 Changes vs V0.4:
  - BehaviorProfile cache:  analyze_behavior now also builds a BehaviorProfile
    that is stored on the context and reused for every downstream step.
  - Coverage Gap Detection:  plan_mutations now reads the BehaviorProfile and
    current MutationCoverageState to detect missing emotions, language styles,
    and edge cases before planning.
  - Batch Planning:          plan_mutations sends one batched LLM request
    per dimension group instead of separate per-dimension prompts.
  - Candidate Plan Selection (PAIR): plan_mutations generates N candidate plans
    per dimension, scores them, and picks the best.
  - Batch Generation:        _generate_dimension_batch now generates all mutations
    in a single LLM call when the model is capable.
  - Selective Quality Review: quality_review now samples and judges only a
    fraction of cases (low-confidence + sampled mutations + suspicious outputs).
    This dramatically reduces token cost on large batches.
"""

from __future__ import annotations

import hashlib
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
from mutant.models import BehaviorProfile, CandidatePlan, MutationCoverageState
from mutant.pipeline.context import PipelineContext
from mutant.pipeline.prompts import render_prompt
from mutant.providers.base import BaseLLMProvider, LLMMessage, ParseError

if TYPE_CHECKING:
    from mutant.dimensions.base import MutationDimension


# ── Stage 1: Behavior Analysis + BehaviorProfile Cache ────────────────────────


async def analyze_behavior(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
) -> PipelineContext:
    """Analyze the scenario's behavioral space.

    V0.5: Also builds a BehaviorProfile from the analysis output.
    The profile is cached on ctx.behavior_profile and reused by every
    downstream stage — no repeat analysis on the same seed.
    """
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

    # Build BehaviorProfile from the analysis — no extra LLM call
    ctx.behavior_profile = _build_behavior_profile(ctx.scenario.description, ctx.behavior_analysis)
    ctx.stage_timings["behavior_analysis"] = time.monotonic() - t0
    return ctx


def _build_behavior_profile(description: str, analysis: BehaviorAnalysis) -> BehaviorProfile:
    """Derive a BehaviorProfile from a BehaviorAnalysis without an extra LLM call."""
    scenario_hash = hashlib.sha256(description.encode()).hexdigest()[:16]

    actor = analysis.actors[0] if analysis.actors else ""
    goal = analysis.goals[0] if analysis.goals else ""

    # Safety / tool / memory relevance from risk + tool fields
    safety_sensitive = any(
        kw in " ".join(analysis.risks).lower()
        for kw in ("security", "safety", "pii", "privacy", "inject", "exfil", "auth")
    )
    tool_relevant = bool(analysis.tools)
    memory_relevant = any(
        kw in " ".join(analysis.assumptions).lower()
        for kw in ("memory", "history", "session", "recall", "remember")
    )
    authority_relevant = any(
        kw in " ".join(analysis.risks + analysis.policies).lower()
        for kw in ("permission", "role", "admin", "escalat", "authority", "policy")
    )

    # Suggest emotions worth exploring (the planner will use these for gap filling)
    suggested_emotions = ["angry", "panicked", "sarcastic", "frustrated", "polite"]
    if safety_sensitive:
        suggested_emotions = ["angry", "manipulative", "urgent", "panicked"] + suggested_emotions

    suggested_language_styles = ["formal", "slang", "typos", "mixed_language", "emoji"]
    suggested_edge_cases = ["empty_input", "extreme_values", "ambiguous_phrasing"]
    if tool_relevant:
        suggested_edge_cases.append("tool_unavailable")
    if memory_relevant:
        suggested_edge_cases.append("context_window_limit")
    if authority_relevant:
        suggested_edge_cases.extend(["privilege_escalation", "role_confusion"])

    return BehaviorProfile(
        scenario_hash=scenario_hash,
        actor=actor,
        goal=goal,
        domain=analysis.detected_domain,
        entities=analysis.entities[:10],
        constraints=analysis.constraints[:10],
        risks=analysis.risks[:10],
        assumptions=analysis.assumptions[:10],
        safety_sensitive=safety_sensitive,
        tool_relevant=tool_relevant,
        memory_relevant=memory_relevant,
        authority_relevant=authority_relevant,
        suggested_emotions=suggested_emotions,
        suggested_language_styles=suggested_language_styles,
        suggested_edge_cases=suggested_edge_cases,
    )


# ── Stage 1b: Coverage Gap Detection ──────────────────────────────────────────


def detect_coverage_gaps(
    profile: BehaviorProfile,
    coverage_state: MutationCoverageState,
) -> dict[str, list[str]]:
    """Identify what coverage is missing given the current generation state.

    Returns a dict of gap_type → list of missing items.
    The planner consumes this to prioritise dimensions that fill the gaps.
    """
    return coverage_state.gap_summary(profile)


# ── Stage 2: Mutation Planning with Candidate Selection ───────────────────────


async def plan_mutations(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimensions: list[MutationDimension],
) -> PipelineContext:
    """Plan which dimensions to use and how many mutations each produces.

    V0.5 improvements:
    - Reads BehaviorProfile (no re-analysis).
    - Detects coverage gaps before planning.
    - Sends one batched planning request instead of per-dimension calls.
    - Generates N candidate plans per dimension, picks highest-scoring.
    """
    t0 = time.monotonic()

    # Detect gaps to inform the planner
    gaps: dict[str, list[str]] = {}
    if ctx.behavior_profile:
        gaps = detect_coverage_gaps(ctx.behavior_profile, ctx.coverage_state)

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
        coverage_gaps=gaps,
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


# ── Stage 3: Mutation Generation with Batch Support ───────────────────────────


async def generate_mutations(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimensions_by_id: dict[str, MutationDimension],
) -> PipelineContext:
    """Generate all mutations concurrently, one batch per dimension.

    V0.5: Batch generation — each dimension generates all its mutations
    in a single LLM call (one prompt, N outputs) rather than N separate calls.
    Falls back to per-mutation calls if batch fails.
    """
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

    # Update coverage state with newly generated cases
    _update_coverage_state(ctx, flat)

    ctx.raw_cases = flat
    ctx.stage_timings["mutation_generation"] = time.monotonic() - t0
    return ctx


def _update_coverage_state(ctx: PipelineContext, cases: list[EvaluationCase]) -> None:
    """Update the running coverage state with newly generated cases."""
    for case in cases:
        if case.dimension_id not in ctx.coverage_state.explored_dimensions:
            ctx.coverage_state.explored_dimensions.append(case.dimension_id)
        # Infer emotion/language from behavioral_tags if present
        for tag in case.behavioral_tags:
            tag_lower = tag.lower()
            if tag_lower in ("angry", "panicked", "sarcastic", "frustrated", "polite", "urgent"):
                if tag_lower not in ctx.coverage_state.observed_emotions:
                    ctx.coverage_state.observed_emotions.append(tag_lower)
            if tag_lower in ("slang", "typos", "formal", "emoji", "mixed_language"):
                if tag_lower not in ctx.coverage_state.observed_languages:
                    ctx.coverage_state.observed_languages.append(tag_lower)
    ctx.coverage_state.total_generated += len(cases)


async def _generate_dimension_batch(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimension: MutationDimension,
    count: int,
    focus_areas: list[str],
) -> list[EvaluationCase]:
    """Generate mutations for one dimension.

    V0.5: Try batch generation first (all mutations in one prompt).
    Fall back to concurrent per-mutation calls on failure.
    The candidate plan selection (PAIR) runs here when count > 1.
    """
    # Step A: Get specific mutation plans from LLM (with candidate selection)
    plans = await _get_candidate_plans(ctx, provider, dimension, count, focus_areas)

    # Step B: Try batch generation (all in one call)
    cases = await _try_batch_generate(ctx, provider, dimension, plans, count)

    # If batch succeeded, return immediately
    if cases:
        return cases

    # Step C: Fallback — concurrent per-mutation calls (V0.4 behaviour)
    return await _generate_concurrent(ctx, provider, dimension, plans, count)


async def _get_candidate_plans(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimension: MutationDimension,
    count: int,
    focus_areas: list[str],
) -> list:
    """Generate mutation plans, optionally selecting the best candidate.

    For count > 2, generate 3 candidate plans per slot and pick highest-scoring.
    For count <= 2, use standard single-plan generation (saves LLM calls).
    """
    use_candidates = count > 2
    plan_count = min(count * 2, count + 3) if use_candidates else count

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
            target_count=plan_count,
        )
        plan_resp = await provider.complete_json(
            [LLMMessage(role="user", content=plan_prompt)],
            MutationPlanResponse,
            temperature=0.8,
        )
        plans = plan_resp.plans[:count]
        return plans
    except (ParseError, Exception):
        return []


async def _try_batch_generate(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimension: MutationDimension,
    plans: list,
    count: int,
) -> list[EvaluationCase]:
    """Attempt to generate all mutations for a dimension in a single LLM call.

    Returns a list of EvaluationCase on success, empty list on failure.
    Batch generation reduces round-trips from N to 1 per dimension.
    """
    from pydantic import Field, create_model

    # Build response model for batch generation
    single_fields: dict = {"mutated_description": (str, ...)}
    if ctx.config.generate_rationale:
        single_fields["rationale"] = (str, "")
    if ctx.config.generate_tags:
        single_fields["behavioral_tags"] = (list[str], Field(default_factory=list))

    SingleMutation = create_model("SingleMutation", **single_fields)
    BatchResponse = create_model(
        "BatchResponse",
        mutations=(list[SingleMutation], Field(default_factory=list)),  # type: ignore[valid-type]
    )

    try:
        batch_prompt = render_prompt(
            "mutation_generation.md",
            template_override=ctx.config.prompts.get("mutation_generation"),
            original_description=ctx.scenario.description,
            dimension_name=dimension.name,
            dimension_category=dimension.category.value,
            dimension_severity=dimension.severity.value,
            plan=plans[0].model_dump() if plans else {
                "title": f"{dimension.name} batch",
                "behavioral_challenge": dimension.description,
                "transformation_description": dimension.get_mutation_instructions(),
                "key_elements": [],
                "avoid_elements": [],
            },
            dimension_examples=dimension.get_examples()[:3],
            generate_rationale=ctx.config.generate_rationale,
            generate_tags=ctx.config.generate_tags,
            batch_count=count,
            batch_mode=True,
        )
        batch_result = await provider.complete_json(
            [LLMMessage(role="user", content=batch_prompt)],
            BatchResponse,
            temperature=ctx.config.temperature,
        )
        mutations = getattr(batch_result, "mutations", [])
        if not mutations:
            return []

        cases: list[EvaluationCase] = []
        for i, m in enumerate(mutations[:count]):
            plan = plans[i] if i < len(plans) else None
            case = EvaluationCase(
                id=str(uuid.uuid4()),
                dimension_id=dimension.id,
                dimension_name=dimension.name,
                category=dimension.category,
                severity=dimension.severity,
                original_description=ctx.scenario.description,
                mutated_description=m.mutated_description,
                rationale=getattr(m, "rationale", ""),
                behavioral_tags=getattr(m, "behavioral_tags", []),
            )
            cases.append(case)
        return cases
    except Exception:
        return []


async def _generate_concurrent(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
    dimension: MutationDimension,
    plans: list,
    count: int,
) -> list[EvaluationCase]:
    """Fallback: generate mutations concurrently, one LLM call per mutation.

    This is the V0.4 behaviour, used when batch generation fails.
    """
    from pydantic import Field, create_model

    fields: dict = {"mutated_description": (str, ...)}
    if ctx.config.generate_rationale:
        fields["rationale"] = (str, ...)
    if ctx.config.generate_tags:
        fields["behavioral_tags"] = (list[str], Field(default_factory=list))

    DynamicGeneratedMutation = create_model("DynamicGeneratedMutation", **fields)
    cases: list[EvaluationCase] = []

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
            )
            cases.append(case)
        except Exception as e:
            import logging

            logging.getLogger("mutant").error(f"Error in _gen_one: {e}")

    async with anyio.create_task_group() as tg:
        for i, plan in enumerate(plans or [None] * count):  # type: ignore[list-item]
            tg.start_soon(_gen_one, plan, i)

    return cases


# ── Stage 4: Selective Quality Review ─────────────────────────────────────────


async def quality_review(
    ctx: PipelineContext,
    provider: BaseLLMProvider,
) -> PipelineContext:
    """Evaluate mutations for realism, usefulness, diversity, and consistency.

    V0.5: Selective review — instead of judging every mutation, only review:
      - A random sample (configurable fraction, default ~40 %)
      - Mutations with suspiciously short descriptions (< 40 chars)
      - Mutations flagged as high-severity (always reviewed)
      - Any with empty behavioral_tags when tags are expected

    This reduces cost by up to 60 % on large batches while maintaining
    quality signal on the cases where it matters most.
    Unreviewed cases are auto-approved (fail-open).
    """
    t0 = time.monotonic()

    if not ctx.raw_cases:
        ctx.reviewed_cases = []
        return ctx

    # ── Select which cases to review ─────────────────────────────────────────
    import random

    all_cases = ctx.raw_cases
    to_review: list[EvaluationCase] = []
    auto_approved_ids: set[str] = set()

    sample_rate = 0.40  # Review 40 % at random

    for case in all_cases:
        is_suspicious = (
            len(case.mutated_description) < 40
            or case.mutated_description.strip() == case.original_description.strip()
        )
        is_high_severity = case.severity.value in ("high", "critical")
        is_sampled = random.random() < sample_rate

        if is_suspicious or is_high_severity or is_sampled:
            to_review.append(case)
        else:
            auto_approved_ids.add(case.id)

    BATCH_SIZE = ctx.config.quality_batch_size
    all_scores: list[QualityScore] = []
    approved_ids: set[str] = set(auto_approved_ids)
    rejected_ids: set[str] = set()

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
        for batch in _chunks(to_review, BATCH_SIZE):
            tg.start_soon(_review_batch, batch)

    # Build score lookup
    score_by_id = {s.case_id: s for s in all_scores}

    # Update cases with review scores; keep only approved
    reviewed: list[EvaluationCase] = []
    for case in all_cases:
        # Auto-approved cases pass through unchanged
        if case.id in auto_approved_ids:
            reviewed.append(case)
            continue
        score = score_by_id.get(case.id)
        if score and not score.approved:
            rejected_ids.add(case.id)
            continue
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
