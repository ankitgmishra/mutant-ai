"""
mutant/core/engine.py — V0.5
==============================
MutationEngine orchestrates the 6-stage pipeline.
Python coordinates; LLMs generate.

V0.5 Changes:
  - BehaviorProfile is built during analysis and cached on PipelineContext.
  - Coverage gap detection feeds the planner with structured gap data.
  - Selective quality review: only ~40% of cases are judged (configurable).
  - Batch generation: dimensions generate all mutations in one prompt.

Pipeline:
  analyze_behavior (+ profile cache) → plan_mutations (gap-aware)
  → generate_mutations (batched) → quality_review (selective)
  → deduplicate → output
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("mutant")

from mutant.core.config import MutationConfig
from mutant.core.mutation import (
    AugmentedDataset,
    EvaluationCase,
    MutationResult,
)
from mutant.core.registry import MutationRegistry
from mutant.core.registry import registry as _default_registry
from mutant.core.scenario import Scenario
from mutant.pipeline.context import PipelineConfig, PipelineContext
from mutant.pipeline.stages import (
    analyze_behavior,
    deduplicate,
    generate_mutations,
    plan_mutations,
)

if TYPE_CHECKING:
    from mutant.cache.base import BaseCache
    from mutant.dimensions.base import MutationDimension
    from mutant.providers.base import BaseLLMProvider

# Keep old name as alias for backward compat
MutationCase = EvaluationCase


class MutationEngine:
    """Orchestrates the V0.4 mutation pipeline.

    All generation is delegated to the LLM. Python only coordinates.

    Example
    -------
    >>> engine = MutationEngine(provider=OpenAIProvider())
    >>> result = await engine.run(scenario, count=20)
    >>> print(result.stats)
    >>> print(result.mutation_plan)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        registry: MutationRegistry | None = None,
        cache: BaseCache | None = None,
    ) -> None:
        self._provider: Any = _CachedProvider(provider, cache) if cache else provider
        self._registry = registry or _default_registry

    async def run(
        self,
        scenario: Scenario,
        config: MutationConfig | None = None,
        **kwargs: Any,
    ) -> MutationResult:
        """Run the full 6-stage mutation pipeline.

        Parameters
        ----------
        scenario : Scenario
        config : MutationConfig | None
            Configuration object. If provided, overrides kwargs.
        **kwargs : Any
            Config parameters if config is not provided.
        """
        if config is None:
            config = MutationConfig(**kwargs)

        if config.verbose:
            logger.info("Analyzing scenario...")

        active_dims = self._resolve_dimensions(
            dimension_ids=config.dimension_ids,
            exclude_dimension_ids=config.exclude_dimension_ids,
            categories=config.categories,
            severities=config.severities,
        )
        [d.id for d in self._registry.all()]

        pipeline_config = PipelineConfig(
            count=config.count,
            concurrency=config.concurrency,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            quality_review=config.quality_review,
            quality_batch_size=config.quality_batch_size,
            deduplicate=config.deduplicate,
            generate_rationale=config.generate_rationale,
            generate_tags=config.generate_tags,
            dimension_ids=[d.id for d in active_dims],
            prompts=config.prompts,
        )
        ctx = PipelineContext(scenario=scenario, config=pipeline_config)
        dims_by_id = {d.id: d for d in active_dims}

        # Stage 1
        ctx = await analyze_behavior(ctx, self._provider)

        if config.verbose:
            logger.info("Planning mutations...")

        # Stage 2
        ctx = await plan_mutations(ctx, self._provider, active_dims)

        if config.verbose:
            logger.info("Generating mutations...")

        # Stage 3
        ctx = await generate_mutations(ctx, self._provider, dims_by_id)

        # Stage 4 — Quality Review
        if config.quality_review:
            if config.verbose:
                logger.info("Reviewing quality...")
            from mutant.pipeline.stages import quality_review as _qr

            ctx = await _qr(ctx, self._provider)
        else:
            ctx.reviewed_cases = ctx.raw_cases

        # Stage 5 — Deduplication
        if config.deduplicate and len(ctx.reviewed_cases) > 1:
            if config.verbose:
                logger.info("Deduplicating...")
            ctx = await deduplicate(ctx, self._provider)
        else:
            ctx.deduplicated_cases = ctx.reviewed_cases

        if config.verbose:
            logger.info("Completed.")

        ctx.final_cases = ctx.output_cases

        return MutationResult(
            cases=ctx.final_cases,
            behavior_analysis=ctx.behavior_analysis,
            mutation_plan=ctx.mutation_plan,
        )

    def _resolve_dimensions(
        self,
        *,
        dimension_ids: list[str] | None,
        exclude_dimension_ids: list[str] | None,
        categories: list[str] | None,
        severities: list[str] | None,
    ) -> list[MutationDimension]:
        pool = self._registry.all()
        if dimension_ids:
            id_set = set(dimension_ids)
            pool = [d for d in pool if d.id in id_set]
        if exclude_dimension_ids:
            exclude_set = set(exclude_dimension_ids)
            pool = [d for d in pool if d.id not in exclude_set]
        if categories:
            cat_set = {c.lower() for c in categories}
            pool = [d for d in pool if d.category.value.lower() in cat_set]
        if severities:
            sev_set = {s.lower() for s in severities}
            pool = [d for d in pool if d.severity.value.lower() in sev_set]
        if not pool:
            raise ValueError(
                "No dimensions match the provided filters. "
                "Check your dimension ids, categories, and severities."
            )
        return pool


# ── Caching Provider Wrapper ───────────────────────────────────────────────────


class _CachedProvider:
    """Transparent provider wrapper that adds response caching."""

    def __init__(self, provider: BaseLLMProvider, cache: BaseCache) -> None:
        self._provider = provider
        self._cache = cache
        self.provider_name = provider.provider_name

    async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        from mutant.cache.base import CacheEntry, CacheKey

        key = CacheKey.from_request(
            provider=self._provider.provider_name,
            model=getattr(self._provider, "model", "unknown"),
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.8),
        )
        cached = await self._cache.get(key)
        if cached:
            from mutant.providers.base import LLMResponse

            return LLMResponse(
                content=cached.content, model="cached", metadata={"cache_hit": True}
            )
        response = await self._provider.complete(messages, **kwargs)
        await self._cache.set(CacheEntry(key=key, content=response.content))
        return response

    async def complete_json(self, messages, schema, **kwargs):  # type: ignore[no-untyped-def]
        response = await self.complete(messages, **kwargs)  # type: ignore[no-untyped-call]
        return self._provider._parse_json(response.content, schema)

    @staticmethod
    def _parse_json(content: str, schema):  # type: ignore[no-untyped-def]
        from mutant.providers.base import BaseLLMProvider

        return BaseLLMProvider._parse_json(content, schema)


async def mutate(
    scenario: Scenario,
    provider: BaseLLMProvider,
    count: int | None = None,
    quality_review: bool | None = None,
    dimensions: list[str] | None = None,
    verbose: bool | None = None,
    generate_rationale: bool | None = None,
    generate_tags: bool | None = None,
    config: MutationConfig | None = None,
    cache: BaseCache | None = None,
    registry: MutationRegistry | None = None,
    **kwargs: Any,
) -> MutationResult:
    """Generate LLM-powered behavioral mutations. Primary public API."""
    engine = MutationEngine(provider=provider, registry=registry, cache=cache)
    if config is None:
        config = MutationConfig(
            count=count if count is not None else 50,
            quality_review=quality_review if quality_review is not None else True,
            dimension_ids=dimensions,
            verbose=verbose if verbose is not None else False,
            generate_rationale=generate_rationale
            if generate_rationale is not None
            else True,
            generate_tags=generate_tags if generate_tags is not None else True,
            **kwargs,
        )
    else:
        # User explicitly passed a config, but we can override with common kwargs if provided
        updates: dict[str, Any] = {}
        if count is not None:
            updates["count"] = count
        if quality_review is not None:
            updates["quality_review"] = quality_review
        if verbose is not None:
            updates["verbose"] = verbose
        if dimensions is not None:
            updates["dimension_ids"] = dimensions
        if generate_rationale is not None:
            updates["generate_rationale"] = generate_rationale
        if generate_tags is not None:
            updates["generate_tags"] = generate_tags
        if updates:
            config = config.model_copy(update=updates)

    return await engine.run(scenario, config=config, **kwargs)


def mutate_sync(
    scenario: Scenario,
    provider: BaseLLMProvider,
    count: int | None = None,
    quality_review: bool | None = None,
    dimensions: list[str] | None = None,
    verbose: bool | None = None,
    generate_rationale: bool | None = None,
    generate_tags: bool | None = None,
    config: MutationConfig | None = None,
    **kwargs: Any,
) -> MutationResult:
    """Synchronous wrapper for mutate."""
    return asyncio.run(
        mutate(
            scenario,
            provider,
            count=count,
            quality_review=quality_review,
            dimensions=dimensions,
            verbose=verbose,
            config=config,
            **kwargs,
        )
    )


async def augment(
    dataset: Sequence[Scenario],
    provider: BaseLLMProvider,
    mutations_per_case: int | None = None,
    quality_review: bool | None = None,
    dimensions: list[str] | None = None,
    verbose: bool | None = None,
    generate_rationale: bool | None = None,
    generate_tags: bool | None = None,
    concurrency: int = 3,
    config: MutationConfig | None = None,
    **kwargs: Any,
) -> AugmentedDataset:
    """Augment an existing dataset by mutating each scenario."""
    import anyio

    if config is None:
        config = MutationConfig(
            count=mutations_per_case if mutations_per_case is not None else 5,
            quality_review=quality_review if quality_review is not None else True,
            dimension_ids=dimensions,
            verbose=verbose if verbose is not None else False,
            generate_rationale=generate_rationale
            if generate_rationale is not None
            else True,
            generate_tags=generate_tags if generate_tags is not None else True,
            **kwargs,
        )
    else:
        updates: dict[str, Any] = {}
        if mutations_per_case is not None:
            updates["count"] = mutations_per_case
        if quality_review is not None:
            updates["quality_review"] = quality_review
        if verbose is not None:
            updates["verbose"] = verbose
        if dimensions is not None:
            updates["dimension_ids"] = dimensions
        if generate_rationale is not None:
            updates["generate_rationale"] = generate_rationale
        if generate_tags is not None:
            updates["generate_tags"] = generate_tags
        if updates:
            config = config.model_copy(update=updates)

    all_results: list[MutationResult | None] = [None for _ in dataset]
    semaphore = anyio.Semaphore(concurrency)

    async def _process_one(
        idx: int, scenario: Scenario, task_id: Any = None, progress: Any = None
    ) -> None:
        async with semaphore:
            try:
                # Disable child verbosity so the progress bar isn't interrupted by logs
                child_config = config.model_copy(update={"verbose": False})
                result = await mutate(scenario, provider=provider, config=child_config)
                all_results[idx] = result
            except Exception as e:
                logger.error(f"Failed to process scenario {idx}: {e}")
            finally:
                if progress is not None and task_id is not None:
                    progress.advance(task_id)

    if config.verbose:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Augmenting dataset...", total=len(dataset)
            )
            async with anyio.create_task_group() as tg:
                for i, scenario in enumerate(dataset):
                    tg.start_soon(_process_one, i, scenario, task_id, progress)
    else:
        async with anyio.create_task_group() as tg:
            for i, scenario in enumerate(dataset):
                tg.start_soon(_process_one, i, scenario, None, None)

    final_cases = []
    for res in all_results:
        if res:
            final_cases.extend(res.cases)

    return AugmentedDataset(cases=final_cases)


def augment_sync(
    dataset: Sequence[Scenario],
    provider: BaseLLMProvider,
    mutations_per_case: int | None = None,
    quality_review: bool | None = None,
    dimensions: list[str] | None = None,
    verbose: bool | None = None,
    generate_rationale: bool | None = None,
    generate_tags: bool | None = None,
    **kwargs: Any,
) -> AugmentedDataset:
    """Synchronous wrapper for dataset augmentation."""
    return asyncio.run(
        augment(
            dataset,
            provider,
            mutations_per_case=mutations_per_case,
            quality_review=quality_review,
            dimensions=dimensions,
            verbose=verbose,
            **kwargs,
        )
    )
