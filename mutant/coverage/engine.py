"""
mutant/coverage/engine.py
==========================
Core coverage analysis engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import anyio
from pydantic import BaseModel, Field

from mutant.core.scenario import Scenario
from mutant.coverage.report import CoverageReport
from mutant.coverage.taxonomy import DEFAULT_TAXONOMY, BehaviorTaxonomy

if TYPE_CHECKING:
    from mutant.providers.base import BaseLLMProvider


class _ScenarioClassification(BaseModel):
    behaviors: list[str] = Field(
        default_factory=list,
        description="List of dimension IDs that are present in this scenario.",
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        default="Medium",
        description="Estimated difficulty of this scenario for an AI agent.",
    )
    requires_tool: bool = Field(
        default=False,
        description="Whether this scenario explicitly requires the agent to use external tools.",
    )
    is_multi_turn: bool = Field(
        default=False,
        description="Whether this scenario represents a multi-turn conversation (vs a single prompt).",
    )
    emotion: Literal[
        "Neutral", "Polite", "Angry", "Frustrated", "Happy", "Panicked", "Other"
    ] = Field(
        default="Neutral",
        description="The primary emotion or tone exhibited by the user in this scenario.",
    )
    language: Literal[
        "Formal", "Informal", "Typos", "Mixed Language", "Emoji", "Other"
    ] = Field(default="Formal", description="The linguistic style of the user prompt.")
    domain: Literal[
        "General Chat", "Coding", "Medical", "Finance", "Legal", "Other"
    ] = Field(
        default="General Chat",
        description="The primary domain or topic of the scenario.",
    )


def _build_prompt(scenario: Scenario, taxonomy: BehaviorTaxonomy) -> str:
    categories = taxonomy.registry.categories()

    blocks = []
    for cat in categories:
        dims = taxonomy.registry.by_category(cat)
        if not dims:
            continue
        blocks.append(f"  {cat.value.title()}:")
        for dim in dims:
            blocks.append(f"    - {dim.id} ({dim.name}): {dim.description}")

    taxonomy_block = "\n".join(blocks)

    return f"""You are a behavioral analyst classifying AI agent test scenarios.

Your task is to identify which behaviors from the taxonomy below are present, and extract metadata.

RULES:
- Only output the exact dimension IDs (e.g. "reasoning.ambiguous") in the behaviors list.
- A behavior is "present" if the scenario meaningfully exhibits that behavior.
- Classify difficulty as Easy, Medium, or Hard.
- Determine if the scenario requires tools, and if it is multi-turn.
- Extract Input Diversity attributes (Emotion, Language style, Domain).

TAXONOMY:
{taxonomy_block}

SCENARIO:
Title: {scenario.title}
Description: {scenario.description}

Return ONLY valid JSON:
{{
  "behaviors": ["list of matching dimension IDs from the taxonomy"],
  "difficulty": "Easy",
  "requires_tool": false,
  "is_multi_turn": false,
  "emotion": "Neutral",
  "language": "Formal",
  "domain": "General Chat"
}}

Return ONLY the JSON. No explanation. No markdown fences."""


async def _classify_one(
    scenario: Scenario,
    provider: BaseLLMProvider,
    taxonomy: BehaviorTaxonomy,
    results: list[_ScenarioClassification | None],
    idx: int,
    semaphore: anyio.abc.ObjectReceiveStream,
) -> None:
    from mutant.providers.base import LLMMessage

    try:
        prompt = _build_prompt(scenario, taxonomy)
        classification = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            _ScenarioClassification,
            temperature=0.1,
        )
        valid = set(taxonomy.all_dimension_ids)
        classification.behaviors = [b for b in classification.behaviors if b in valid]
        results[idx] = classification
    except Exception:
        results[idx] = None


async def coverage(
    dataset: Sequence[Scenario],
    provider: BaseLLMProvider,
    taxonomy: BehaviorTaxonomy | None = None,
    concurrency: int = 5,
    verbose: bool = False,
) -> CoverageReport:
    """Analyze the behavioral coverage and return a CoverageReport."""
    if taxonomy is None:
        taxonomy = DEFAULT_TAXONOMY

    if not dataset:
        return CoverageReport(taxonomy, {}, {}, 0)

    n = len(dataset)
    results: list[_ScenarioClassification | None] = [None for _ in range(n)]

    import logging

    _logger = logging.getLogger("mutant")
    if verbose:
        _logger.info(f"Analysing {n} scenarios...")

    sem = anyio.Semaphore(concurrency)

    async def _run_one(idx: int, scenario: Scenario) -> None:
        async with sem:
            await _classify_one(scenario, provider, taxonomy, results, idx, sem)

    async with anyio.create_task_group() as tg:
        for i, scenario in enumerate(dataset):
            tg.start_soon(_run_one, i, scenario)

    from collections import Counter

    freq: Counter[str] = Counter()

    difficulty_counts = Counter()
    tool_counts = Counter()
    turn_counts = Counter()
    emotion_counts = Counter()
    language_counts = Counter()
    domain_counts = Counter()

    seen_descriptions = set()
    duplicates = 0

    for i, res in enumerate(results):
        desc = dataset[i].description.lower().strip()
        if desc in seen_descriptions:
            duplicates += 1
        else:
            seen_descriptions.add(desc)

        if res is not None:
            for behavior in res.behaviors:
                freq[behavior] += 1

            difficulty_counts[res.difficulty] += 1
            emotion_counts[res.emotion] += 1
            language_counts[res.language] += 1
            domain_counts[res.domain] += 1

            if res.requires_tool:
                tool_counts["Requires Tool"] += 1
            else:
                tool_counts["No Tool"] += 1

            if res.is_multi_turn:
                turn_counts["Multi Turn"] += 1
            else:
                turn_counts["Single Turn"] += 1

    full_freq = {b: freq.get(b, 0) for b in taxonomy.all_dimension_ids}

    metadata = {
        "difficulty": dict(difficulty_counts),
        "tools": dict(tool_counts),
        "turns": dict(turn_counts),
        "emotion": dict(emotion_counts),
        "language": dict(language_counts),
        "domain": dict(domain_counts),
        "duplicates": duplicates,
        "unique": n - duplicates,
    }

    if verbose:
        _logger.info("Analysis complete.")

    return CoverageReport(taxonomy, full_freq, metadata, n)
