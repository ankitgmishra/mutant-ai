"""
mutant/eval/metrics/base.py
=============================
Abstract base class for all evaluation metrics.

Three metric families:
  - Deterministic: Pure computation, no LLM needed (regex, exact match, JSON valid).
  - LLM-as-Judge: Uses an LLM provider to score (correctness, faithfulness, etc.).
  - Composite: Combines multiple child metrics into a single aggregate score.

Every metric implements ``async score(test_case) → MetricResult``.
The base class provides ``batch_score()`` with concurrency control.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import anyio

from mutant.eval.types import MetricResult, TestCase, Verdict

if TYPE_CHECKING:
    pass


class Metric(ABC):
    """Abstract base class for all evaluation metrics.

    Subclasses must implement ``score()``. The base class handles:
      - Threshold comparison and verdict assignment.
      - Batch scoring with concurrency control.
      - Required field validation before scoring.

    Parameters
    ----------
    name : str
        Human-readable metric name (e.g. ``"Correctness"``).
    threshold : float
        Minimum score to pass (0.0 – 1.0). Default 0.5.

    Example
    -------
    >>> class MyMetric(Metric):
    ...     def __init__(self):
    ...         super().__init__(name="MyMetric", threshold=0.7)
    ...     async def score(self, test_case):
    ...         return self._result(score=0.9, reason="Looks good")
    """

    # Override in subclasses to declare which TestCase fields are required.
    # Score will return SKIP if any of these are None/empty.
    required_fields: tuple[str, ...] = ("input", "actual_output")

    def __init__(
        self,
        name: str,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.threshold = max(0.0, min(1.0, threshold))

    @abstractmethod
    async def score(self, test_case: TestCase) -> MetricResult:
        """Score a single test case.

        Subclasses MUST call ``self._result(score=..., reason=...)``
        to build the MetricResult with automatic threshold comparison.

        Parameters
        ----------
        test_case : TestCase
            The test case to evaluate.

        Returns
        -------
        MetricResult
            Score, verdict, and explanation.
        """

    async def batch_score(
        self,
        test_cases: list[TestCase],
        concurrency: int = 10,
    ) -> list[MetricResult]:
        """Score multiple test cases with concurrency control.

        Parameters
        ----------
        test_cases : list[TestCase]
            Test cases to evaluate.
        concurrency : int
            Max concurrent scoring tasks.

        Returns
        -------
        list[MetricResult]
            Results in the same order as input test cases.
        """
        results: list[MetricResult | None] = [None] * len(test_cases)
        semaphore = anyio.Semaphore(concurrency)

        async def _score_one(idx: int, tc: TestCase) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.safe_score(tc)
                except Exception as e:
                    results[idx] = self._error_result(str(e))

        async with anyio.create_task_group() as tg:
            for i, tc in enumerate(test_cases):
                tg.start_soon(_score_one, i, tc)

        return [r for r in results if r is not None]

    async def safe_score(self, test_case: TestCase) -> MetricResult:
        """Score with validation and error handling.

        Checks required fields, catches exceptions, and returns
        appropriate SKIP or ERROR results.
        """
        # Check required fields
        for field in self.required_fields:
            value = test_case
            for part in field.split("."):
                if value is None:
                    break
                value = getattr(value, part, None)
                
            if value is None or (isinstance(value, (str, list, dict)) and not value):
                return MetricResult(
                    metric_name=self.name,
                    score=0.0,
                    threshold=self.threshold,
                    passed=False,
                    verdict=Verdict.SKIP,
                    reason=f"Required field '{field}' is missing or empty.",
                )

        try:
            return await self.score(test_case)
        except Exception as e:
            return self._error_result(str(e))

    # ── Result builders ──────────────────────────────────────────────────────

    def _result(
        self,
        score: float,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MetricResult:
        """Build a MetricResult with automatic threshold comparison."""
        clamped = max(0.0, min(1.0, score))
        passed = clamped >= self.threshold
        return MetricResult(
            metric_name=self.name,
            score=clamped,
            threshold=self.threshold,
            passed=passed,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            reason=reason,
            metadata=metadata or {},
        )

    def _error_result(self, error_msg: str) -> MetricResult:
        """Build an ERROR MetricResult."""
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            threshold=self.threshold,
            passed=False,
            verdict=Verdict.ERROR,
            reason=f"Metric error: {error_msg}",
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"
