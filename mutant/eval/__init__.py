"""
mutant/eval/__init__.py
========================
Mutant Evaluation Engine — mutation-driven LLM/AI evaluation.

This module provides a full evaluation framework that integrates with
Mutant's mutation engine to enable:

  Generate adversarial mutations → Run against target → Evaluate → Report

Usage
-----
Traditional evaluation (like DeepEval/Ragas)::

    from mutant.eval import EvalSuite, Correctness, Toxicity, TestCase

    suite = EvalSuite(metrics=[Correctness(provider), Toxicity(provider)])
    report = await suite.run(test_cases)
    report.display()

Mutant evaluation (the differentiator)::

    from mutant import mutate, Scenario
    from mutant.eval import EvalSuite, Correctness, Faithfulness
    from mutant.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o-mini")
    scenario = Scenario(title="Refund", description="Customer requests refund...")
    mutations = await mutate(scenario, provider=provider, count=50)

    suite = EvalSuite(metrics=[Correctness(provider), Faithfulness(provider)])
    report = await suite.run_against(target=my_chatbot, mutations=mutations)
    report.display()
"""

from mutant.eval.metrics import (
    AnswerRelevancy,
    BiasDetection,
    Coherence,
    Contains,
    ContextPrecision,
    ContextRecall,
    Correctness,
    CustomCallableMetric,
    CustomLLMMetric,
    CustomRubric,
    ExactMatch,
    Faithfulness,
    InstructionBoundary,
    JsonValid,
    LatencyCheck,
    LengthConstraint,
    LLMJudgeMetric,
    Metric,
    NotEmpty,
    PromptInjectionResistance,
    RAGContextLeakage,
    RefusalDetection,
    RegexMatch,
    Relevance,
    SensitiveDataLeakage,
    SystemPromptLeakage,
    ToolArgumentSafety,
    Toxicity,
)
from mutant.eval.gate import GateResult, assert_security_gate, run_security_gate
from mutant.eval.report import EvalReport
from mutant.eval.suite import EvalSuite, evaluate, evaluate_against
from mutant.eval.types import EvalResult, MetricResult, SecurityTestCase, TestCase, Verdict

__all__ = [
    # Core types
    "TestCase",
    "SecurityTestCase",
    "MetricResult",
    "EvalResult",
    "Verdict",
    # Suite & Report
    "EvalSuite", "evaluate", "evaluate_against",
    "EvalReport",
    # Base
    "Metric",
    "LLMJudgeMetric",
    # Deterministic metrics
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JsonValid",
    "LengthConstraint",
    "NotEmpty",
    "LatencyCheck",
    # LLM-as-Judge metrics
    "Correctness",
    "Faithfulness",
    "Relevance",
    "Coherence",
    "ContextPrecision",
    "ContextRecall",
    "Toxicity",
    "BiasDetection",
    "AnswerRelevancy",
    "RefusalDetection",
    # Security metrics (new)
    "SensitiveDataLeakage",
    "PromptInjectionResistance",
    "InstructionBoundary",
    "SystemPromptLeakage",
    "RAGContextLeakage",
    "ToolArgumentSafety",
    # Security gate — the model-free CI path
    "run_security_gate",
    "assert_security_gate",
    "GateResult",
    # Custom metrics
    "CustomRubric",
    "CustomLLMMetric",
    "CustomCallableMetric",
]
