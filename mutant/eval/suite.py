"""
mutant/eval/suite.py
=====================
EvalSuite — the evaluation orchestrator.

Two modes:
  1. ``run(test_cases)`` — Evaluate pre-built test cases (like DeepEval/Ragas).
  2. ``run_against(target, mutations)`` — Mutant's killer feature:
     Take mutation output, run each against a target, evaluate, and report.

The suite handles concurrency, progress reporting, error recovery,
and result aggregation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import anyio

from mutant.eval.metrics.base import Metric
from mutant.eval.report import EvalReport
from mutant.eval.types import EvalResult, TestCase, Verdict

if TYPE_CHECKING:
    from mutant.core.mutation import MutationResult

logger = logging.getLogger("mutant.eval")

# Type alias matching redteam's TargetFn — but we accept it locally
# to avoid tight coupling. Any async str → str callable works, and a target may also
# return a TestCase to attach observables (tool calls, retrieved context, secrets) that
# a plain string cannot carry.
TargetFn = Callable[[str], Awaitable[str | TestCase]]


class EvalSuite:
    """Evaluation suite — runs metrics against test cases or live targets.

    Parameters
    ----------
    metrics : list[Metric]
        Metrics to evaluate. Each test case is scored by every metric.
    concurrency : int
        Max concurrent target calls / metric evaluations.
    verbose : bool
        Enable progress logging.

    Example — Traditional evaluation
    ---------------------------------
    >>> suite = EvalSuite(metrics=[Correctness(provider), Toxicity(provider)])
    >>> report = await suite.run(test_cases)
    >>> report.display()

    Example — Mutant evaluation (the killer feature)
    -------------------------------------------------
    >>> mutations = await mutate(scenario, provider=provider, count=50)
    >>> suite = EvalSuite(metrics=[Correctness(provider), Toxicity(provider)])
    >>> report = await suite.run_against(target=my_chatbot, mutations=mutations)
    >>> report.display()
    """

    def __init__(
        self,
        metrics: list[Metric],
        *,
        concurrency: int = 10,
        verbose: bool = False,
    ) -> None:
        if not metrics:
            raise ValueError("EvalSuite requires at least one metric.")
        self.metrics = metrics
        self.concurrency = concurrency
        self.verbose = verbose

    async def run(
        self,
        test_cases: list[TestCase],
    ) -> EvalReport:
        """Evaluate pre-built test cases against all metrics.

        Parameters
        ----------
        test_cases : list[TestCase]
            Test cases with ``actual_output`` already filled in.

        Returns
        -------
        EvalReport
            Aggregated evaluation results.
        """
        t0 = time.monotonic()

        if self.verbose:
            logger.info(
                f"Evaluating {len(test_cases)} test cases "
                f"with {len(self.metrics)} metrics..."
            )

        results = await self._evaluate_all(test_cases)
        duration = time.monotonic() - t0

        if self.verbose:
            passed = sum(1 for r in results if r.passed)
            logger.info(
                f"Evaluation complete: {passed}/{len(results)} passed "
                f"in {duration:.1f}s"
            )

        return EvalReport(
            results=results,
            metrics_used=[m.name for m in self.metrics],
            duration_seconds=duration,
        )

    async def run_against(
        self,
        target: TargetFn,
        mutations: MutationResult | list[TestCase],
        *,
        expected_output_fn: Callable[[str], str | Awaitable[str]] | None = None,
        context_fn: Callable[[str], list[str] | Awaitable[list[str]]] | None = None,
    ) -> EvalReport:
        """Mutant's killer feature: generate → run → evaluate → report.

        Takes mutation output (from ``mutate()``), runs each mutation
        against the target system, evaluates the response with all metrics,
        and produces a comprehensive evaluation report.

        Parameters
        ----------
        target : TargetFn
            Async callable ``str → str`` — the system under test.
        mutations : MutationResult | list[TestCase]
            Either a ``MutationResult`` from ``mutate()`` or pre-built TestCases.
        expected_output_fn : callable | None
            Optional function to generate expected output for each input.
            ``(input: str) → str``. Can be sync or async.
        context_fn : callable | None
            Optional function to provide ground-truth context for each input.
            ``(input: str) → list[str]``. Can be sync or async.

        Returns
        -------
        EvalReport
            Comprehensive evaluation report with per-mutation,
            per-metric, and per-dimension breakdowns.

        Example
        -------
        >>> async def my_chatbot(msg: str) -> str:
        ...     return "I can help with that!"
        ...
        >>> mutations = await mutate(scenario, provider=provider, count=20)
        >>> suite = EvalSuite(metrics=[Correctness(provider)])
        >>> report = await suite.run_against(target=my_chatbot, mutations=mutations)
        >>> report.display()
        """
        t0 = time.monotonic()

        # Convert MutationResult to TestCases
        test_cases = self._prepare_test_cases(mutations)

        if self.verbose:
            logger.info(
                f"Running {len(test_cases)} mutations against target..."
            )

        # Run all mutations against the target
        test_cases = await self._run_target(
            target, test_cases, expected_output_fn, context_fn
        )

        if self.verbose:
            logger.info(
                f"Target responses collected. Evaluating with "
                f"{len(self.metrics)} metrics..."
            )

        # Evaluate all responses
        results = await self._evaluate_all(test_cases)
        duration = time.monotonic() - t0

        if self.verbose:
            passed = sum(1 for r in results if r.passed)
            logger.info(
                f"Evaluation complete: {passed}/{len(results)} passed "
                f"in {duration:.1f}s"
            )

        return EvalReport(
            results=results,
            metrics_used=[m.name for m in self.metrics],
            duration_seconds=duration,
        )

    # ── Internal: Run target ─────────────────────────────────────────────────

    async def _run_target(
        self,
        target: TargetFn,
        test_cases: list[TestCase],
        expected_output_fn: Callable | None,
        context_fn: Callable | None,
    ) -> list[TestCase]:
        """Run all test cases against the target with concurrency control."""
        semaphore = anyio.Semaphore(self.concurrency)
        enriched: list[TestCase] = [None] * len(test_cases)  # type: ignore[list-item]

        async def _call_one(idx: int, tc: TestCase) -> None:
            async with semaphore:
                start = time.monotonic()
                try:
                    response = await target(tc.input)
                    latency = (time.monotonic() - start) * 1000

                    if isinstance(response, TestCase):
                        # A target may return a TestCase to attach observable data
                        # (tool calls, retrieved context, sensitive strings) that a
                        # plain string cannot carry. Merge those fields onto the
                        # prepared case instead of replacing it, so the mutation
                        # provenance (dimension/severity) survives for reporting.
                        #
                        # ``input`` is deliberately NOT overridable: the question
                        # belongs to the harness, and the metrics must always be
                        # scored against the mutation that was actually sent.
                        overrides: dict[str, Any] = {
                            field: getattr(response, field)
                            for field in response.model_fields_set
                            if field not in {"id", "input", "mutation", "latency_ms"}
                        }
                        if "metadata" in overrides:
                            overrides["metadata"] = {
                                **tc.metadata,
                                **overrides["metadata"],
                            }
                        overrides["latency_ms"] = latency
                        if response.mutation is None and tc.mutation is not None:
                            overrides["mutation"] = tc.mutation
                        enriched[idx] = tc.model_copy(update=overrides)
                    else:
                        updates: dict[str, Any] = {
                            "actual_output": response,
                            "latency_ms": latency,
                        }

                        # Optional: generate expected output
                        if expected_output_fn:
                            import inspect
                            if inspect.iscoroutinefunction(expected_output_fn):
                                expected = await expected_output_fn(tc.input)
                            else:
                                expected = expected_output_fn(tc.input)
                            updates["expected_output"] = expected

                        # Optional: generate context
                        if context_fn:
                            import inspect
                            if inspect.iscoroutinefunction(context_fn):
                                ctx = await context_fn(tc.input)
                            else:
                                ctx = context_fn(tc.input)
                            updates["context"] = ctx

                        enriched[idx] = tc.model_copy(update=updates)

                except Exception as e:
                    latency = (time.monotonic() - start) * 1000
                    enriched[idx] = tc.model_copy(
                        update={
                            "actual_output": f"[TARGET ERROR: {e}]",
                            "latency_ms": latency,
                        }
                    )

        async with anyio.create_task_group() as tg:
            for i, tc in enumerate(test_cases):
                tg.start_soon(_call_one, i, tc)

        return enriched

    # ── Internal: Evaluate ───────────────────────────────────────────────────

    async def _evaluate_all(
        self,
        test_cases: list[TestCase],
    ) -> list[EvalResult]:
        """Evaluate all test cases against all metrics."""
        results: list[EvalResult | None] = [None] * len(test_cases)
        semaphore = anyio.Semaphore(self.concurrency)

        async def _eval_one(idx: int, tc: TestCase) -> None:
            async with semaphore:
                t0 = time.monotonic()
                metric_results: dict[str, Any] = {}

                for metric in self.metrics:
                    try:
                        result = await metric.safe_score(tc)
                        metric_results[metric.name] = result
                    except Exception as e:
                        metric_results[metric.name] = metric._error_result(str(e))

                # A case only counts as passed when at least one metric reached a
                # decisive PASS and none failed. A case scored entirely UNKNOWN
                # (missing trace/tool/context data) is inconclusive, not a pass —
                # otherwise a security suite that observed nothing would report 100%.
                all_passed = all(r.passed for r in metric_results.values())
                any_decisive_pass = any(
                    r.verdict == Verdict.PASS for r in metric_results.values()
                )
                duration = time.monotonic() - t0

                results[idx] = EvalResult(
                    test_case=tc,
                    metric_results=metric_results,
                    passed=all_passed and any_decisive_pass,
                    duration_seconds=duration,
                )

        async with anyio.create_task_group() as tg:
            for i, tc in enumerate(test_cases):
                tg.start_soon(_eval_one, i, tc)

        return [r for r in results if r is not None]

    # ── Internal: Prepare test cases ─────────────────────────────────────────

    @staticmethod
    def _prepare_test_cases(
        source: Any,  # MutationResult | list[TestCase]
    ) -> list[TestCase]:
        """Convert various input formats to a list of TestCase."""
        # Already TestCases
        if isinstance(source, list) and source and isinstance(source[0], TestCase):
            return source

        # MutationResult or AugmentedDataset — has .cases
        if hasattr(source, "cases"):
            return [
                TestCase.from_evaluation_case(case)
                for case in source.cases
            ]

        # List of EvaluationCases
        if isinstance(source, list):
            return [TestCase.from_evaluation_case(case) for case in source]

        raise TypeError(
            f"Cannot convert {type(source).__name__} to test cases. "
            "Pass a MutationResult, AugmentedDataset, list[TestCase], "
            "or list[EvaluationCase]."
        )

async def evaluate(
    test_cases: list[TestCase],
    metrics: list[Metric],
    *,
    concurrency: int = 10,
    verbose: bool = True,
) -> EvalReport:
    """Evaluate pre-built test cases against a list of metrics.
    
    This is a convenience function equivalent to:
        suite = EvalSuite(metrics)
        return await suite.run(test_cases)
    """
    suite = EvalSuite(metrics=metrics, concurrency=concurrency, verbose=verbose)
    return await suite.run(test_cases)

async def evaluate_against(
    target: TargetFn,
    mutations: Any,
    metrics: list[Metric],
    *,
    expected_output_fn: Callable[[str], str | Awaitable[str]] | None = None,
    context_fn: Callable[[str], list[str] | Awaitable[list[str]]] | None = None,
    concurrency: int = 10,
    verbose: bool = True,
) -> EvalReport:
    """Run generated mutations against a live target and evaluate.
    
    This is a convenience function equivalent to:
        suite = EvalSuite(metrics)
        return await suite.run_against(target, mutations)
    """
    suite = EvalSuite(metrics=metrics, concurrency=concurrency, verbose=verbose)
    return await suite.run_against(
        target,
        mutations,
        expected_output_fn=expected_output_fn,
        context_fn=context_fn,
    )
