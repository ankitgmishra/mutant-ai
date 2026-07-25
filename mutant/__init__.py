"""
Mutant — LLM-native behavioral dataset engineering for AI agents.

Mutant automatically generates realistic behavioral variations ("mutations")
from a single scenario using an LLM as the creative engine, so developers
can discover how their AI agents fail before users do.

Quickstart
----------
>>> import asyncio
>>> from mutant import mutate, Scenario
>>> from mutant.providers import OpenAIProvider
>>>
>>> provider = OpenAIProvider(model="gpt-4o-mini")
>>> scenario = Scenario(
...     title="Refund Request",
...     description="Customer bought a laptop and requests a refund after 10 days.",
... )
>>> cases = asyncio.run(mutate(scenario, provider=provider, count=20))
>>> for case in cases:
...     print(f"[{case.severity.value}] {case.dimension_name}")
...     print(f"  {case.mutated_description[:80]}...")
"""

import mutant.dimensions  # noqa: F401 — auto-registers all built-in dimensions
from mutant.core.config import MutationConfig
from mutant.core.engine import (
    MutationEngine,
    augment,
    augment_sync,
    mutate,
    mutate_sync,
)
from mutant.core.mutation import (
    AugmentedDataset,
    BehaviorAnalysis,
    EvaluationCase,
    MutationCase,  # backward-compat alias
    MutationCategory,
    MutationResult,
    MutationSeverity,
    QualityReviewResult,
)
from mutant.core.registry import MutationRegistry, registry
from mutant.core.scenario import Scenario
from mutant.coverage import DEFAULT_TAXONOMY, BehaviorTaxonomy, CoverageReport, coverage
from mutant.datasets.io import (
    load_csv,
    load_dataframe,
    load_huggingface,
    load_json,
    load_jsonl,
)
from mutant.exceptions import MutantError, ParseError, ProviderError
from mutant.redteam import (
    RedTeamReport,
    TargetProfile,
    Transcript,
    red_team,
    red_team_sync,
)

__all__ = [
    "DEFAULT_TAXONOMY",
    "AugmentedDataset",
    "BehaviorAnalysis",
    "BehaviorTaxonomy",
    "CoverageReport",
    "EvaluationCase",
    # Exceptions
    "MutantError",
    "MutationCase",  # backward-compat
    "MutationCategory",
    "MutationConfig",
    # Engine
    "MutationEngine",
    # Plugin system
    "MutationRegistry",
    "MutationResult",
    "MutationSeverity",
    "ParseError",
    "ProviderError",
    "QualityReviewResult",
    # Red Team API
    "RedTeamReport",
    # Core types
    "Scenario",
    "TargetProfile",
    "Transcript",
    "augment",
    "augment_sync",
    # Coverage API
    "coverage",
    # Datasets API
    "load_csv",
    "load_dataframe",
    "load_huggingface",
    "load_json",
    "load_jsonl",
    # Primary API
    "mutate",
    "mutate_sync",
    # Red Team API
    "red_team",
    "red_team_sync",
    "registry",
]

__version__ = "0.5.0"
__author__ = "Ankit Mishra"
