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


# ── Contexts ─────────────────────────────────────────────────────────────────


class RAGContext(BaseModel):
    """Context for RAG evaluations."""
    context: list[str] = Field(
        default_factory=list,
        description="Ground-truth context documents (e.g. for faithfulness).",
    )
    retrieval_context: list[str] = Field(
        default_factory=list,
        description="Documents actually retrieved by the RAG system.",
    )


class AgentContext(BaseModel):
    """Context for Agent evaluations."""
    tools_called: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls made by the target agent.",
    )
    # Could add trajectory, tool_results, etc.


class ConversationContext(BaseModel):
    """Context for Multi-turn evaluations."""
    messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns [{role, content}, ...].",
    )


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

    The core unit that flows through the evaluation pipeline. It uses a
    composable architecture where specialized fields are grouped into contexts.

    Parameters
    ----------
    input : str | None
        The prompt or message sent to the target system.
    actual_output : str | None
        The target's response. ``None`` until the target is called.
    expected_output : str | None
        Optional golden/reference answer for correctness checking.
    rag : RAGContext | None
        Context for RAG evaluations.
    agent : AgentContext | None
        Context for agent/tool evaluations.
    conversation : ConversationContext | None
        Context for multi-turn evaluations.
    mutation : MutationContext | None
        Context for mutation provenance.
    metadata : dict
        Free-form metadata.

    Example
    -------
    >>> tc = TestCase(
    ...     input="What is your refund policy?",
    ...     actual_output="Our refund policy allows returns within 30 days.",
    ...     expected_output="Returns accepted within 30 days of purchase.",
    ... )
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str | None = Field(default=None, description="Prompt sent to the target.")
    actual_output: str | None = Field(
        default=None, description="Target's response. None until evaluated."
    )
    expected_output: str | None = Field(
        default=None, description="Optional golden/reference answer."
    )

    # Composable contexts
    rag: RAGContext | None = Field(default=None, description="RAG specific fields.")
    agent: AgentContext | None = Field(default=None, description="Agent specific fields.")
    conversation: ConversationContext | None = Field(default=None, description="Multi-turn specific fields.")
    mutation: MutationContext | None = Field(default=None, description="Mutation specific fields.")

    # Free-form
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata."
    )

    # Timing
    latency_ms: float | None = Field(
        default=None, description="Response latency in milliseconds."
    )

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
        """Build a TestCase from a Mutant EvaluationCase.

        This is the primary bridge between the mutation engine and the
        evaluation engine.

        Parameters
        ----------
        case : EvaluationCase
            The mutation case (from ``mutate()``).
        actual_output : str | None
            The target's response (filled in by ``EvalSuite.run_against()``).
        expected_output : str | None
            Optional golden answer.
        context : list[str] | None
            Ground-truth context for RAG evaluation.
        retrieval_context : list[str] | None
            Retrieved documents from the RAG system.
        """
        rag_context = None
        if context or retrieval_context:
            rag_context = RAGContext(
                context=context or [],
                retrieval_context=retrieval_context or []
            )

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
            rag=rag_context,
            mutation=mutation_context,
        )


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
