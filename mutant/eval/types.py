"""
mutant/eval/types.py
=====================
Core data types for the evaluation engine.

TestCase is the central unit — it carries the input (possibly from a mutation),
the target's actual output, and optional context for RAG/agent evaluation.
MetricResult holds the score from a single metric on a single test case.
EvalResult aggregates all metric results for one test case.

Design:
  - TestCase is NOT frozen — it's built incrementally (input first, then output
    after running against a target).
  - MetricResult IS frozen — once scored, it doesn't change.
  - EvalResult IS frozen — a completed evaluation snapshot.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# ── Verdict ──────────────────────────────────────────────────────────────────


class Verdict(StrEnum):
    """Outcome of a metric evaluation."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    UNKNOWN = "unknown"


# ── Contexts ─────────────────────────────────────────────────────────────────

class MutationContext(BaseModel):
    """Context for Mutation provenance."""
    original_input: str | None = Field(
        default=None,
        description="Pre-mutation input, if this came from mutate().",
    )
    dimension_id: str | None = Field(default=None, description="Source mutation dimension.")
    dimension_name: str | None = Field(default=None, description="Human-readable dimension name.")
    category: str | None = Field(default=None, description="Mutation category.")
    severity: str | None = Field(default=None, description="Mutation severity.")
    mutation_metadata: dict[str, Any] = Field(default_factory=dict, description="Rationle, behavioral tags, etc.")


# ── TestCase ─────────────────────────────────────────────────────────────────


class TestCase(BaseModel):
    """A single evaluation test case.

    Parameters
    ----------
    input : str | None
        The prompt or message sent to the target system.
    actual_output : str | None
        The target's response. ``None`` until the target is called.
    expected_output : str | None
        Optional golden/reference answer for correctness checking.
    context : list[str] | None
        Optional reference/ground-truth context.
    retrieval_context : list[str] | None
        Context actually retrieved by the application at runtime.
    tools_called : list[dict[str, Any]] | None
        Tools actually called during execution.
    expected_tools : list[dict[str, Any]] | None
        Expected agent tools (golden trajectory).
    available_tools : list[dict[str, Any]] | None
        Tools available to the agent.
    metadata : dict
        Free-form metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str | None = Field(default=None, description="Prompt sent to the target.")
    actual_output: str | None = Field(
        default=None, description="Target's response. None until evaluated."
    )
    expected_output: str | None = Field(
        default=None, description="Optional golden/reference answer."
    )

    # RAG
    context: list[str] | None = Field(default=None, description="Reference/ground-truth context.")
    retrieval_context: list[str] | None = Field(default=None, description="Context actually retrieved.")

    # Agent
    expected_tools: list[dict[str, Any]] | None = Field(default=None, description="Expected tools.")
    tools_called: list[dict[str, Any]] | None = Field(default=None, description="Tools actually called.")
    available_tools: list[dict[str, Any]] | None = Field(default=None, description="Available tools.")

    # Multi-turn
    messages: list[dict[str, str]] | None = Field(default=None, description="Prior conversation turns.")

    # Mutation provenance
    mutation: MutationContext | None = Field(default=None, description="Mutation specific fields.")

    # Free-form
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata."
    )

    # Timing
    latency_ms: float | None = Field(
        default=None, description="Response latency in milliseconds."
    )

    # Security-specific (optional — enables Security Evaluation without requiring traces)
    sensitive_data: list[str] | None = Field(
        default=None, description="Sensitive strings that must not appear in output (API keys, secrets, PII, etc.)."
    )
    expected_behavior: str | None = Field(
        default=None, description="Expected behavior for instruction boundary checks."
    )
    system_prompt: str | None = Field(
        default=None, description="Hidden/system prompt text to check for leakage."
    )
    # Alias for tool_calls to match SecurityTestCase spec (tool_calls vs tools_called)
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Alias for tools_called — structured tool calls for ToolArgumentSafety."
    )

    def get_tool_calls(self) -> list[dict[str, Any]] | None:
        """Return observable tool calls, checking both tool_calls and tools_called."""
        if self.tool_calls is not None:
            return self.tool_calls
        return self.tools_called

    def get_retrieved_context(self) -> list[str] | None:
        """Return retrieved context, checking retrieval_context and context."""
        if self.retrieval_context is not None:
            return self.retrieval_context
        return self.context

    @classmethod
    def from_evaluation_case(
        cls,
        case: Any,  # EvaluationCase — avoid circular import
        *,
        actual_output: str | None = None,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
    ) -> TestCase:
        mutation_context = MutationContext(
            original_input=case.original_description,
            dimension_id=case.dimension_id,
            dimension_name=case.dimension_name,
            category=case.category.value if hasattr(case.category, "value") else str(case.category),
            severity=case.severity.value if hasattr(case.severity, "value") else str(case.severity),
            mutation_metadata={
                "rationale": getattr(case, "rationale", ""),
                "behavioral_tags": getattr(case, "behavioral_tags", []),
                "source_case_id": getattr(case, "id", ""),
            }
        )

        return cls(
            input=case.mutated_description,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            mutation=mutation_context,
        )


# Alias for security test cases — TestCase now supports all security fields
# Users can do: SecurityTestCase(input=..., sensitive_data=[...], tool_calls=[...])
SecurityTestCase = TestCase


# ── MetricResult ─────────────────────────────────────────────────────────────


class MetricResult(BaseModel):
    """Result of evaluating a single metric on a single test case.

    Parameters
    ----------
    metric_name : str
        Name of the metric that produced this result.
    score : float
        Numeric score (0.0 – 1.0). Higher is always better.
    threshold : float
        The minimum score for passing.
    passed : bool
        Whether the score meets or exceeds the threshold.
    verdict : Verdict
        Pass/Fail/Error/Skip.
    reason : str
        Human-readable explanation of the score.
    metadata : dict
        Extra info (e.g. LLM judge reasoning, matched patterns, etc.).
    """

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    verdict: Verdict = Verdict.PASS
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# ── EvalResult ───────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Aggregated evaluation result for a single test case across all metrics.

    Parameters
    ----------
    test_case : TestCase
        The evaluated test case (with actual_output filled in).
    metric_results : dict[str, MetricResult]
        Metric name → result mapping.
    passed : bool
        True only if ALL metrics passed.
    duration_seconds : float
        Wall-clock time for evaluating this test case.
    """

    test_case: TestCase
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)
    passed: bool = True
    duration_seconds: float = 0.0

    @property
    def status(self) -> Verdict:
        """Decisive verdict for this case across all metrics.

        Unlike ``passed`` (which only asks "did any metric fail?"), ``status``
        distinguishes a case that was actually verified from one where every
        metric lacked the data to judge:

        - ``FAIL``    — at least one metric failed or errored.
        - ``UNKNOWN`` — every metric was UNKNOWN/SKIP (nothing to verify).
        - ``PASS``    — at least one metric passed and none failed.
        """
        verdicts = [r.verdict for r in self.metric_results.values()]
        if not verdicts:
            return Verdict.UNKNOWN
        if any(v in (Verdict.FAIL, Verdict.ERROR) for v in verdicts):
            return Verdict.FAIL
        if all(v in (Verdict.UNKNOWN, Verdict.SKIP) for v in verdicts):
            return Verdict.UNKNOWN
        return Verdict.PASS

    @property
    def is_inconclusive(self) -> bool:
        """True when no metric could reach a decisive verdict."""
        return self.status == Verdict.UNKNOWN

    @property
    def failed_metrics(self) -> list[str]:
        """Return names of metrics that failed."""
        return [name for name, r in self.metric_results.items() if not r.passed]

    @property
    def avg_score(self) -> float:
        """Average score across all metrics."""
        scores = [r.score for r in self.metric_results.values()]
        return sum(scores) / max(len(scores), 1)

    def score_for(self, metric_name: str) -> float | None:
        """Get the score for a specific metric, or None if not evaluated."""
        r = self.metric_results.get(metric_name)
        return r.score if r else None
