"""
mutant/eval/metrics/custom.py
===============================
User-defined custom evaluation metrics.

CustomRubric: Define evaluation criteria in natural language,
and the LLM judges against your specific rubric.

CustomLLMMetric: Provide a full custom prompt template for
maximum control over LLM-as-judge evaluation.

CustomCallableMetric: Wrap any sync/async function as a metric.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from mutant.eval.metrics.llm_judge import LLMJudgeMetric
from mutant.eval.metrics.base import Metric
from mutant.eval.types import MetricResult, TestCase

if TYPE_CHECKING:
    from mutant.providers.base import BaseLLMProvider


class CustomRubric(LLMJudgeMetric):
    """LLM-as-judge metric with user-defined evaluation criteria.

    The simplest way to create a custom evaluation metric. Provide
    your criteria as natural language, and the LLM judges against it.

    Parameters
    ----------
    provider : BaseLLMProvider
        LLM provider for judge calls.
    name : str
        Name for your metric.
    criteria : str
        Natural language description of what constitutes a good response.
        Be as specific as possible.
    scoring_rubric : str | None
        Optional custom scoring guide. If not provided, uses default
        0.0-1.0 scale description.
    threshold : float
        Minimum passing score. Default 0.5.

    Example
    -------
    >>> metric = CustomRubric(
    ...     provider=my_provider,
    ...     name="Empathy",
    ...     criteria="The response should acknowledge the user's frustration, "
    ...              "show understanding of their situation, and offer concrete "
    ...              "next steps. Avoid dismissive language or generic platitudes.",
    ... )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        name: str = "CustomRubric",
        criteria: str,
        scoring_rubric: str | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, name=name, threshold=threshold, **kwargs)
        self.criteria = criteria
        self.scoring_rubric = scoring_rubric

    def _build_prompt(self, test_case: TestCase) -> str:
        scoring_section = self.scoring_rubric or """## Scoring
- 1.0 = Fully meets all criteria
- 0.7-0.9 = Meets most criteria with minor issues
- 0.4-0.6 = Partially meets criteria
- 0.1-0.3 = Mostly fails to meet criteria
- 0.0 = Completely fails all criteria"""

        context_section = ""
        rag = test_case.rag
        if rag and rag.context:
            context_text = "\n---\n".join(rag.context)
            context_section = f"\n## Context\n{context_text}\n"

        expected_section = ""
        if test_case.expected_output:
            expected_section = f"\n## Expected Output (Reference)\n{test_case.expected_output}\n"

        return f"""You are an expert evaluation judge. Evaluate the response using the following custom criteria.

## Evaluation Criteria: {self.name}
{self.criteria}

## Input
{test_case.input}

## Actual Output
{test_case.actual_output}
{context_section}{expected_section}
{scoring_section}

Return a JSON object with:
- "score": float (0.0 to 1.0)
- "verdict": "pass" or "fail"
- "reason": brief explanation of how well the output meets the criteria"""


class CustomLLMMetric(LLMJudgeMetric):
    """Fully custom LLM-as-judge metric with user-provided prompt template.

    - ``input``: The test case input
    - ``actual_output``: The target's response
    - ``expected_output``: Optional reference answer
    - ``context``: List of context documents (if RAG context present)
    - ``retrieval_context``: List of retrieved documents
    - ``metadata``: Test case metadata dict
    - ``conversation``: ConversationContext (if multi-turn)
    - ``agent``: AgentContext (if agent evaluation)

    Parameters
    ----------
    provider : BaseLLMProvider
        LLM provider for judge calls.
    name : str
        Name for your metric.
    prompt_template : str
        Full prompt template string. Must instruct the LLM to return
        JSON with "score" (0.0-1.0), "verdict" ("pass"/"fail"), and "reason".
    threshold : float
        Minimum passing score. Default 0.5.

    Example
    -------
    >>> metric = CustomLLMMetric(
    ...     provider=my_provider,
    ...     name="DomainAccuracy",
    ...     prompt_template=\"\"\"
    ...     Evaluate this medical response for clinical accuracy.
    ...
    ...     Question: {{ input }}
    ...     Response: {{ actual_output }}
    ...
    ...     Score 1.0 for clinically accurate, 0.0 for dangerous misinformation.
    ...     Return JSON: {"score": float, "verdict": "pass"/"fail", "reason": str}
    ...     \"\"\",
    ... )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        name: str = "CustomLLMMetric",
        prompt_template: str,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, name=name, threshold=threshold, **kwargs)
        self.prompt_template = prompt_template

    def _build_prompt(self, test_case: TestCase) -> str:
        from jinja2 import Template

        template = Template(self.prompt_template)
        return template.render(
            input=test_case.input,
            actual_output=test_case.actual_output or "",
            expected_output=test_case.expected_output or "",
            context=test_case.rag.context if test_case.rag else [],
            retrieval_context=test_case.rag.retrieval_context if test_case.rag else [],
            metadata=test_case.metadata,
            conversation=test_case.conversation,
            agent=test_case.agent,
        )


class CustomCallableMetric(Metric):
    """Wrap any Python function as a Mutant evaluation metric.

    The callable receives a TestCase and returns a float score (0.0–1.0).
    Can be sync or async.

    Parameters
    ----------
    name : str
        Name for your metric.
    scoring_fn : callable
        A function ``(TestCase) -> float`` or ``async (TestCase) -> float``.
        Must return a float in [0.0, 1.0].
    threshold : float
        Minimum passing score. Default 0.5.

    Example
    -------
    >>> def word_count_score(tc: TestCase) -> float:
    ...     words = len((tc.actual_output or "").split())
    ...     return min(1.0, words / 100)  # Prefer longer responses
    ...
    >>> metric = CustomCallableMetric(
    ...     name="WordCount",
    ...     scoring_fn=word_count_score,
    ...     threshold=0.3,
    ... )
    """

    def __init__(
        self,
        *,
        name: str = "CustomCallable",
        scoring_fn: Callable[[TestCase], float | Awaitable[float]],
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, threshold=threshold, **kwargs)
        self.scoring_fn = scoring_fn
        self._is_async = inspect.iscoroutinefunction(scoring_fn)

    async def score(self, test_case: TestCase) -> MetricResult:
        try:
            if self._is_async:
                raw_score = await self.scoring_fn(test_case)
            else:
                raw_score = self.scoring_fn(test_case)

            score = max(0.0, min(1.0, float(raw_score)))
            return self._result(score=score, reason=f"Custom function returned {score:.3f}")
        except Exception as e:
            return self._error_result(f"Custom function raised: {e}")
