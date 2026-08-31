"""
mutant/eval/metrics/__init__.py
================================
Public API for all evaluation metrics.
"""

from mutant.eval.metrics.base import Metric
from mutant.eval.metrics.custom import (
    CustomCallableMetric,
    CustomLLMMetric,
    CustomRubric,
)
from mutant.eval.metrics.deterministic import (
    Contains,
    ExactMatch,
    JsonValid,
    LatencyCheck,
    LengthConstraint,
    NotEmpty,
    RegexMatch,
)
from mutant.eval.metrics.llm_judge import (
    AnswerRelevancy,
    BiasDetection,
    Coherence,
    ContextPrecision,
    ContextRecall,
    Correctness,
    Faithfulness,
    LLMJudgeMetric,
    Relevance,
    RefusalDetection,
    Toxicity,
    ToolSelection,
    ToolArgumentCorrectness,
    ToolCallOrder,
    TaskCompletion,
)

__all__ = [
    # Base
    "Metric",
    "LLMJudgeMetric",
    # Deterministic
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JsonValid",
    "LengthConstraint",
    "NotEmpty",
    "LatencyCheck",
    # LLM-as-Judge
    "Correctness",
    "Faithfulness",
    "Relevance",
    "Coherence",
    "Toxicity",
    "BiasDetection",
    "ContextPrecision",
    "ContextRecall",
    "AnswerRelevancy",
    "RefusalDetection",
    # Agent Judges
    "ToolSelection",
    "ToolArgumentCorrectness",
    "ToolCallOrder",
    "TaskCompletion",
    # Custom
    "CustomRubric",
    "CustomLLMMetric",
    "CustomCallableMetric",
]
