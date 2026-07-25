"""Behavioral coverage analysis for evaluation datasets."""

from mutant.coverage.engine import coverage
from mutant.coverage.report import CoverageReport
from mutant.coverage.taxonomy import DEFAULT_TAXONOMY, BehaviorTaxonomy

__all__ = [
    "DEFAULT_TAXONOMY",
    "BehaviorTaxonomy",
    "CoverageReport",
    "coverage",
]
