"""tests/test_eval.py — Comprehensive tests for the evaluation engine."""

from __future__ import annotations

import json
from typing import TypeVar

import pytest
from pydantic import BaseModel

from mutant.eval.types import EvalResult, MetricResult, TestCase, Verdict
from mutant.eval.metrics.base import Metric
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
    _JudgeVerdict,
)
from mutant.eval.metrics.custom import (
    CustomCallableMetric,
    CustomLLMMetric,
    CustomRubric,
)
from mutant.eval.suite import EvalSuite
from mutant.eval.report import EvalReport

T = TypeVar("T", bound=BaseModel)


# ── Mock Provider for LLM-as-Judge Tests ────────────────────────────────────


class MockJudgeProvider:
    """Mock provider that returns configurable judge verdicts."""

    provider_name = "mock_judge"

    def __init__(self, score: float = 0.8, verdict: str = "pass", reason: str = "Good."):
        self._score = score
        self._verdict = verdict
        self._reason = reason
        self.call_count = 0

    async def complete(self, messages, *, temperature=0.8, max_tokens=4096):
        from mutant.providers.base import LLMResponse
        self.call_count += 1
        return LLMResponse(
            content=json.dumps({
                "score": self._score,
                "verdict": self._verdict,
                "reason": self._reason,
            }),
            model="mock",
        )

    async def complete_json(self, messages, schema, *, temperature=0.7, max_tokens=4096, max_retries=3):
        self.call_count += 1
        if schema is _JudgeVerdict:
            return _JudgeVerdict(
                chain_of_thought="Step 1: Mock reasoning.",
                score=self._score,
                verdict=self._verdict,
                reason=self._reason,
            )
        return schema.model_validate({})


# ── Test Helpers ────────────────────────────────────────────────────────────


def make_test_case(**kwargs) -> TestCase:
    """Create a test case with sensible defaults."""
    from mutant.eval.types import MutationContext
    
    mutation_kwargs = {}
    for key in ["dimension_id", "dimension_name", "severity", "category", "original_input"]:
        if key in kwargs:
            mutation_kwargs[key] = kwargs.pop(key)
            
    defaults = {
        "input": "What is the refund policy?",
        "actual_output": "You can return items within 30 days.",
        "expected_output": "Items can be returned within 30 days of purchase.",
    }
    defaults.update(kwargs)
    
    tc = TestCase(**defaults)
    if mutation_kwargs:
        tc.mutation = MutationContext(**mutation_kwargs)
        
    return tc


# ══════════════════════════════════════════════════════════════════════════════
# TestCase
# ══════════════════════════════════════════════════════════════════════════════


class TestTestCase:
    def test_create_basic(self):
        tc = TestCase(input="Hello", actual_output="Hi there!")
        assert tc.input == "Hello"
        assert tc.actual_output == "Hi there!"
        assert tc.expected_output is None
        assert tc.context is None
        assert tc.retrieval_context is None
        assert tc.expected_tools is None
        assert tc.tools_called is None
        assert tc.messages is None
        assert tc.id  # UUID generated

    def test_create_from_evaluation_case(self):
        from mutant.core.mutation import EvaluationCase, MutationCategory, MutationSeverity

        case = EvaluationCase(
            id="test-123",
            dimension_id="emotion.angry",
            dimension_name="Angry Customer",
            category=MutationCategory.EMOTION,
            severity=MutationSeverity.HIGH,
            original_description="Customer requests refund.",
            mutated_description="I am FURIOUS! Give me my money back NOW!",
            rationale="Tests de-escalation.",
            behavioral_tags=["anger", "urgency"],
        )
        tc = TestCase.from_evaluation_case(
            case,
            actual_output="I understand your frustration...",
        )
        assert tc.input == "I am FURIOUS! Give me my money back NOW!"
        assert tc.actual_output == "I understand your frustration..."
        assert tc.mutation.original_input == "Customer requests refund."
        assert tc.mutation.dimension_id == "emotion.angry"
        assert tc.mutation.dimension_name == "Angry Customer"
        assert tc.mutation.category == "emotion"
        assert tc.mutation.severity == "high"
        assert tc.mutation.mutation_metadata["source_case_id"] == "test-123"

    def test_create_with_context(self):
        tc = TestCase(
            input="What is X?",
            actual_output="X is a thing.",
            context=["Document 1 about X.", "Document 2 about X."],
            retrieval_context=["Retrieved doc about X."]
        )
        assert len(tc.context) == 2
        assert len(tc.retrieval_context) == 1

    def test_create_with_conversation(self):
        tc = TestCase(
            input="And what about Y?",
            actual_output="Y is related to X.",
            messages=[
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is a thing."},
            ]
        )
        assert len(tc.messages) == 2

    def test_metadata(self):
        tc = TestCase(
            input="Hello",
            actual_output="Hi",
            metadata={"custom_key": "custom_value"},
        )
        assert tc.metadata["custom_key"] == "custom_value"

    def test_latency(self):
        tc = TestCase(input="Hello", actual_output="Hi", latency_ms=150.5)
        assert tc.latency_ms == 150.5


# ══════════════════════════════════════════════════════════════════════════════
# MetricResult
# ══════════════════════════════════════════════════════════════════════════════


class TestMetricResult:
    def test_create_pass(self):
        mr = MetricResult(
            metric_name="Test",
            score=0.8,
            threshold=0.5,
            passed=True,
            verdict=Verdict.PASS,
            reason="Good result.",
        )
        assert mr.passed is True
        assert mr.verdict == Verdict.PASS

    def test_create_fail(self):
        mr = MetricResult(
            metric_name="Test",
            score=0.3,
            threshold=0.5,
            passed=False,
            verdict=Verdict.FAIL,
            reason="Below threshold.",
        )
        assert mr.passed is False
        assert mr.verdict == Verdict.FAIL

    def test_frozen(self):
        mr = MetricResult(
            metric_name="Test", score=0.5, threshold=0.5,
            passed=True, verdict=Verdict.PASS,
        )
        with pytest.raises(Exception):
            mr.score = 0.9  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# EvalResult
# ══════════════════════════════════════════════════════════════════════════════


class TestEvalResult:
    def test_failed_metrics(self):
        tc = make_test_case()
        er = EvalResult(
            test_case=tc,
            metric_results={
                "A": MetricResult(metric_name="A", score=0.9, threshold=0.5, passed=True, verdict=Verdict.PASS),
                "B": MetricResult(metric_name="B", score=0.2, threshold=0.5, passed=False, verdict=Verdict.FAIL),
            },
            passed=False,
        )
        assert er.failed_metrics == ["B"]
        assert er.avg_score == pytest.approx(0.55)

    def test_score_for(self):
        tc = make_test_case()
        er = EvalResult(
            test_case=tc,
            metric_results={
                "A": MetricResult(metric_name="A", score=0.7, threshold=0.5, passed=True, verdict=Verdict.PASS),
            },
        )
        assert er.score_for("A") == 0.7
        assert er.score_for("Nonexistent") is None


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestExactMatch:
    @pytest.mark.asyncio
    async def test_exact_match_pass(self):
        m = ExactMatch()
        tc = make_test_case(actual_output="hello world", expected_output="hello world")
        result = await m.safe_score(tc)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_fail(self):
        m = ExactMatch()
        tc = make_test_case(actual_output="hello world", expected_output="goodbye world")
        result = await m.safe_score(tc)
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        m = ExactMatch(case_sensitive=False)
        tc = make_test_case(actual_output="Hello World", expected_output="hello world")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_strip_whitespace(self):
        m = ExactMatch(strip_whitespace=True)
        tc = make_test_case(actual_output="  hello  ", expected_output="hello")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_missing_expected(self):
        m = ExactMatch()
        tc = make_test_case(expected_output=None)
        result = await m.safe_score(tc)
        assert result.verdict == Verdict.SKIP


class TestContains:
    @pytest.mark.asyncio
    async def test_all_present(self):
        m = Contains(substrings=["refund", "30 days"])
        tc = make_test_case(actual_output="You can get a refund within 30 days.")
        result = await m.safe_score(tc)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_some_missing(self):
        m = Contains(substrings=["refund", "90 days"], threshold=0.5)
        tc = make_test_case(actual_output="You can get a refund within 30 days.")
        result = await m.safe_score(tc)
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_any_mode(self):
        m = Contains(substrings=["refund", "warranty"], require_all=False)
        tc = make_test_case(actual_output="Your refund has been processed.")
        result = await m.safe_score(tc)
        assert result.score == 1.0


class TestRegexMatch:
    @pytest.mark.asyncio
    async def test_partial_match(self):
        m = RegexMatch(pattern=r"\d{3}-\d{4}")
        tc = make_test_case(actual_output="Call us at 555-1234 for help.")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_match(self):
        m = RegexMatch(pattern=r"\d{3}-\d{4}")
        tc = make_test_case(actual_output="No phone number here.")
        result = await m.safe_score(tc)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_full_match(self):
        m = RegexMatch(pattern=r"\d+", full_match=True)
        tc = make_test_case(actual_output="12345")
        result = await m.safe_score(tc)
        assert result.passed is True


class TestJsonValid:
    @pytest.mark.asyncio
    async def test_valid_json(self):
        m = JsonValid()
        tc = make_test_case(actual_output='{"name": "John", "age": 30}')
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        m = JsonValid()
        tc = make_test_case(actual_output="not json at all")
        result = await m.safe_score(tc)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_required_keys(self):
        m = JsonValid(required_keys=["name", "age"])
        tc = make_test_case(actual_output='{"name": "John"}')
        result = await m.safe_score(tc)
        assert result.score == 0.5


class TestLengthConstraint:
    @pytest.mark.asyncio
    async def test_within_bounds(self):
        m = LengthConstraint(min_length=5, max_length=100)
        tc = make_test_case(actual_output="This is a valid response.")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_too_short(self):
        m = LengthConstraint(min_length=100)
        tc = make_test_case(actual_output="Short")
        result = await m.safe_score(tc)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_word_count(self):
        m = LengthConstraint(min_length=3, max_length=10, unit="words")
        tc = make_test_case(actual_output="one two three four five")
        result = await m.safe_score(tc)
        assert result.passed is True


class TestNotEmpty:
    @pytest.mark.asyncio
    async def test_non_empty(self):
        m = NotEmpty()
        tc = make_test_case(actual_output="Something here.")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_empty(self):
        m = NotEmpty()
        tc = make_test_case(actual_output="   ")
        result = await m.safe_score(tc)
        assert result.passed is False


class TestLatencyCheck:
    @pytest.mark.asyncio
    async def test_within_limit(self):
        m = LatencyCheck(max_latency_ms=1000)
        tc = make_test_case(latency_ms=500)
        result = await m.safe_score(tc)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_over_limit(self):
        m = LatencyCheck(max_latency_ms=1000)
        tc = make_test_case(latency_ms=1500)
        result = await m.safe_score(tc)
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_no_latency(self):
        m = LatencyCheck(max_latency_ms=1000)
        tc = make_test_case(latency_ms=None)
        result = await m.safe_score(tc)
        assert result.passed is True  # Auto-pass


# ══════════════════════════════════════════════════════════════════════════════
# LLM-as-Judge Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMJudgeMetrics:
    @pytest.mark.asyncio
    async def test_correctness(self):
        provider = MockJudgeProvider(score=0.9, reason="Correct answer.")
        m = Correctness(provider)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True
        assert result.score == 0.9
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_correctness_fail(self):
        provider = MockJudgeProvider(score=0.2, verdict="fail", reason="Wrong.")
        m = Correctness(provider, threshold=0.5)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is False
        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_faithfulness(self):
        provider = MockJudgeProvider(score=0.85, reason="Faithful to context.")
        m = Faithfulness(provider)
        tc = make_test_case(
            context=["Returns are allowed within 30 days of purchase."],
        )
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_faithfulness_missing_context(self):
        provider = MockJudgeProvider()
        m = Faithfulness(provider)
        tc = make_test_case(context=[])
        result = await m.safe_score(tc)
        assert result.verdict == Verdict.SKIP  # context is required

    @pytest.mark.asyncio
    async def test_relevance(self):
        provider = MockJudgeProvider(score=0.95, reason="Highly relevant.")
        m = Relevance(provider)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_coherence(self):
        provider = MockJudgeProvider(score=0.7, reason="Mostly coherent.")
        m = Coherence(provider)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_toxicity(self):
        provider = MockJudgeProvider(score=1.0, reason="No toxicity detected.")
        m = Toxicity(provider)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_bias_detection(self):
        provider = MockJudgeProvider(score=0.9, reason="No bias found.")
        m = BiasDetection(provider)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_answer_relevancy(self):
        provider = MockJudgeProvider(score=0.8, reason="Good use of context.")
        m = AnswerRelevancy(provider)
        tc = make_test_case(retrieval_context=["Retrieved document about refunds."])
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_answer_relevancy_missing_context(self):
        provider = MockJudgeProvider()
        m = AnswerRelevancy(provider)
        tc = make_test_case(retrieval_context=[])
        result = await m.safe_score(tc)
        assert result.verdict == Verdict.SKIP

    @pytest.mark.asyncio
    async def test_context_precision(self):
        provider = MockJudgeProvider(score=0.9, reason="Relevant items ranked high.")
        m = ContextPrecision(provider)
        tc = make_test_case(retrieval_context=["Rank 1: Relevant", "Rank 2: Irrelevant"])
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_context_recall(self):
        provider = MockJudgeProvider(score=0.8, reason="Most facts present in context.")
        m = ContextRecall(provider)
        tc = make_test_case(retrieval_context=["Context with facts"])
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_refusal_detection(self):
        provider = MockJudgeProvider(score=0.9, reason="Properly refused.")
        m = RefusalDetection(provider, should_refuse=True)
        tc = make_test_case(actual_output="I cannot help with that.")
        result = await m.safe_score(tc)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_missing_actual_output(self):
        provider = MockJudgeProvider()
        m = Correctness(provider)
        tc = TestCase(input="Hello", actual_output=None)
        result = await m.safe_score(tc)
        assert result.verdict == Verdict.SKIP


# ══════════════════════════════════════════════════════════════════════════════
# Custom Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestCustomMetrics:
    @pytest.mark.asyncio
    async def test_custom_rubric(self):
        provider = MockJudgeProvider(score=0.75, reason="Meets most criteria.")
        m = CustomRubric(
            provider,
            name="Empathy",
            criteria="Response should show empathy and understanding.",
        )
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.passed is True
        assert result.score == 0.75

    @pytest.mark.asyncio
    async def test_custom_llm_metric(self):
        provider = MockJudgeProvider(score=0.6, reason="Acceptable.")
        m = CustomLLMMetric(
            provider,
            name="CustomMetric",
            prompt_template="Evaluate: {{ input }}\nResponse: {{ actual_output }}\nReturn JSON.",
        )
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.score == 0.6

    @pytest.mark.asyncio
    async def test_custom_callable_sync(self):
        def my_scorer(tc: TestCase) -> float:
            return 0.8 if tc.actual_output else 0.0

        m = CustomCallableMetric(name="MySyncMetric", scoring_fn=my_scorer)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_custom_callable_async(self):
        async def my_async_scorer(tc: TestCase) -> float:
            return 0.9

        m = CustomCallableMetric(name="MyAsyncMetric", scoring_fn=my_async_scorer)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_custom_callable_error(self):
        def bad_scorer(tc: TestCase) -> float:
            raise ValueError("Something went wrong!")

        m = CustomCallableMetric(name="BadMetric", scoring_fn=bad_scorer)
        tc = make_test_case()
        result = await m.safe_score(tc)
        assert result.verdict == Verdict.ERROR


# ══════════════════════════════════════════════════════════════════════════════
# Metric Base — Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchScoring:
    @pytest.mark.asyncio
    async def test_batch_score(self):
        m = NotEmpty()
        tcs = [
            make_test_case(actual_output="Hello"),
            make_test_case(actual_output="World"),
            make_test_case(actual_output="   "),
        ]
        results = await m.batch_score(tcs, concurrency=2)
        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is True
        assert results[2].passed is False


# ══════════════════════════════════════════════════════════════════════════════
# EvalSuite
# ══════════════════════════════════════════════════════════════════════════════


class TestEvalSuite:
    @pytest.mark.asyncio
    async def test_run_basic(self):
        suite = EvalSuite(metrics=[NotEmpty(), LengthConstraint(min_length=3)])
        tcs = [
            make_test_case(actual_output="Good response."),
            make_test_case(actual_output="OK"),
        ]
        report = await suite.run(tcs)
        assert report.total_cases == 2
        assert report.total_passed >= 1

    @pytest.mark.asyncio
    async def test_run_against_target(self):
        async def echo_target(msg: str) -> str:
            return f"Echo: {msg}"

        suite = EvalSuite(metrics=[NotEmpty()])
        tcs = [
            TestCase(input="Hello"),
            TestCase(input="World"),
        ]
        report = await suite.run_against(target=echo_target, mutations=tcs)
        assert report.total_cases == 2
        assert report.total_passed == 2

    @pytest.mark.asyncio
    async def test_run_against_with_mutations(self):
        from mutant.core.mutation import EvaluationCase, MutationCategory, MutationSeverity, MutationResult

        async def mock_bot(msg: str) -> str:
            return "I'll help you with that."

        cases = [
            EvaluationCase(
                id="1",
                dimension_id="emotion.angry",
                dimension_name="Angry Customer",
                category=MutationCategory.EMOTION,
                severity=MutationSeverity.HIGH,
                original_description="Customer requests refund.",
                mutated_description="I am FURIOUS!",
            ),
            EvaluationCase(
                id="2",
                dimension_id="context.missing_information",
                dimension_name="Missing Information",
                category=MutationCategory.CONTEXT,
                severity=MutationSeverity.MEDIUM,
                original_description="Customer requests refund.",
                mutated_description="I want a refund for... something.",
            ),
        ]
        mutations = MutationResult(cases=cases)

        suite = EvalSuite(metrics=[NotEmpty()])
        report = await suite.run_against(target=mock_bot, mutations=mutations)
        assert report.total_cases == 2
        assert report.total_passed == 2
        # Check mutation metadata propagated
        assert report.results[0].test_case.mutation.dimension_id is not None

    @pytest.mark.asyncio
    async def test_run_against_target_error(self):
        async def failing_target(msg: str) -> str:
            raise RuntimeError("Target crashed!")

        suite = EvalSuite(metrics=[NotEmpty()])
        tcs = [TestCase(input="Hello")]
        report = await suite.run_against(target=failing_target, mutations=tcs)
        assert report.total_cases == 1
        # Target error is captured, not raised
        assert "[TARGET ERROR:" in (report.results[0].test_case.actual_output or "")

    def test_empty_metrics_raises(self):
        with pytest.raises(ValueError, match="at least one metric"):
            EvalSuite(metrics=[])


# ══════════════════════════════════════════════════════════════════════════════
# EvalReport
# ══════════════════════════════════════════════════════════════════════════════


class TestEvalReport:
    def _make_report(self) -> EvalReport:
        tcs = [
            make_test_case(
                dimension_id="emotion.angry",
                dimension_name="Angry Customer",
                severity="high",
            ),
            make_test_case(
                dimension_id="emotion.angry",
                dimension_name="Angry Customer",
                severity="high",
            ),
            make_test_case(
                dimension_id="context.missing_information",
                dimension_name="Missing Information",
                severity="medium",
            ),
        ]
        results = [
            EvalResult(
                test_case=tcs[0],
                metric_results={
                    "Correctness": MetricResult(metric_name="Correctness", score=0.9, threshold=0.5, passed=True, verdict=Verdict.PASS, reason="Good."),
                    "Toxicity": MetricResult(metric_name="Toxicity", score=1.0, threshold=0.7, passed=True, verdict=Verdict.PASS, reason="Clean."),
                },
                passed=True,
            ),
            EvalResult(
                test_case=tcs[1],
                metric_results={
                    "Correctness": MetricResult(metric_name="Correctness", score=0.3, threshold=0.5, passed=False, verdict=Verdict.FAIL, reason="Wrong."),
                    "Toxicity": MetricResult(metric_name="Toxicity", score=0.9, threshold=0.7, passed=True, verdict=Verdict.PASS, reason="Clean."),
                },
                passed=False,
            ),
            EvalResult(
                test_case=tcs[2],
                metric_results={
                    "Correctness": MetricResult(metric_name="Correctness", score=0.7, threshold=0.5, passed=True, verdict=Verdict.PASS, reason="OK."),
                    "Toxicity": MetricResult(metric_name="Toxicity", score=0.5, threshold=0.7, passed=False, verdict=Verdict.FAIL, reason="Mildly toxic."),
                },
                passed=False,
            ),
        ]
        return EvalReport(
            results=results,
            metrics_used=["Correctness", "Toxicity"],
            duration_seconds=1.5,
        )

    def test_counts(self):
        report = self._make_report()
        assert report.total_cases == 3
        assert report.total_passed == 1
        assert report.total_failed == 2

    def test_overall_pass_rate(self):
        report = self._make_report()
        assert report.overall_pass_rate == pytest.approx(1 / 3)

    def test_metric_summaries(self):
        report = self._make_report()
        summaries = {ms.name: ms for ms in report.metric_summaries}
        assert "Correctness" in summaries
        assert "Toxicity" in summaries
        assert summaries["Correctness"].total_evaluated == 3
        assert summaries["Correctness"].total_passed == 2
        assert summaries["Correctness"].total_failed == 1

    def test_dimension_breakdowns(self):
        report = self._make_report()
        dims = {db.dimension_id: db for db in report.dimension_breakdowns}
        assert "emotion.angry" in dims
        assert "context.missing_information" in dims
        assert dims["emotion.angry"].total_cases == 2
        assert dims["emotion.angry"].pass_rate == 0.5

    def test_severity_breakdown(self):
        report = self._make_report()
        sev = report.severity_breakdown
        assert "high" in sev
        assert "medium" in sev
        assert sev["high"]["total"] == 2

    def test_failed_results(self):
        report = self._make_report()
        assert len(report.failed_results) == 2

    def test_summary_text(self):
        report = self._make_report()
        text = report.summary()
        assert "Mutant Evaluation Report" in text
        assert "3" in text

    def test_display_no_crash(self):
        report = self._make_report()
        report.display()  # Should not raise

    def test_to_json(self, tmp_path):
        report = self._make_report()
        path = str(tmp_path / "report.json")
        report.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_cases"] == 3
        assert len(data["results"]) == 3

    def test_to_markdown(self, tmp_path):
        report = self._make_report()
        path = str(tmp_path / "report.md")
        report.to_markdown(path)
        with open(path) as f:
            content = f.read()
        assert "# Mutant Evaluation Report" in content


# ══════════════════════════════════════════════════════════════════════════════
# Integration: imports from mutant top-level
# ══════════════════════════════════════════════════════════════════════════════


class TestTopLevelImports:
    def test_import_eval_suite(self):
        from mutant import EvalSuite
        assert EvalSuite is not None

    def test_import_test_case(self):
        from mutant import TestCase
        assert TestCase is not None

    def test_import_eval_report(self):
        from mutant import EvalReport
        assert EvalReport is not None

    def test_import_from_eval_package(self):
        from mutant.eval import (
            Correctness,
            EvalSuite,
            Faithfulness,
            Metric,
            TestCase,
            Toxicity,
        )
        assert all([Correctness, EvalSuite, Faithfulness, Metric, TestCase, Toxicity])
