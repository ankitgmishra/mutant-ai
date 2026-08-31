"""
mutant/eval/metrics/deterministic.py
======================================
Deterministic metrics — pure computation, no LLM calls.

Fast, cheap, and reproducible. Use for structural validation,
format checks, and quantitative constraints.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mutant.eval.metrics.base import Metric
from mutant.eval.types import MetricResult, TestCase


class ExactMatch(Metric):
    """Passes if ``actual_output`` exactly equals ``expected_output``.

    Parameters
    ----------
    case_sensitive : bool
        Whether comparison is case-sensitive. Default True.
    strip_whitespace : bool
        Whether to strip leading/trailing whitespace. Default True.
    """

    required_fields = ("input", "actual_output", "expected_output")

    def __init__(
        self,
        *,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="ExactMatch", threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        if self.strip_whitespace:
            actual = actual.strip()
            expected = expected.strip()

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        match = actual == expected
        return self._result(
            score=1.0 if match else 0.0,
            reason="Exact match." if match else "Output does not match expected.",
        )


class Contains(Metric):
    """Passes if ``actual_output`` contains all specified substrings.

    Parameters
    ----------
    substrings : list[str]
        Substrings that must all be present.
    case_sensitive : bool
        Whether matching is case-sensitive. Default False.
    require_all : bool
        If True (default), ALL substrings must be present.
        If False, at least one must be present.
    """

    required_fields = ("input", "actual_output")

    def __init__(
        self,
        substrings: list[str],
        *,
        case_sensitive: bool = False,
        require_all: bool = True,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="Contains", threshold=threshold, **kwargs)
        self.substrings = substrings
        self.case_sensitive = case_sensitive
        self.require_all = require_all

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = test_case.actual_output or ""
        if not self.case_sensitive:
            actual = actual.lower()

        found = []
        missing = []
        for sub in self.substrings:
            target = sub if self.case_sensitive else sub.lower()
            if target in actual:
                found.append(sub)
            else:
                missing.append(sub)

        if self.require_all:
            score = 1.0 if not missing else len(found) / max(len(self.substrings), 1)
        else:
            score = 1.0 if found else 0.0

        reason = f"Found: {found}." if found else "None of the expected substrings found."
        if missing:
            reason += f" Missing: {missing}."

        return self._result(score=score, reason=reason)


class RegexMatch(Metric):
    """Passes if ``actual_output`` matches a regular expression pattern.

    Parameters
    ----------
    pattern : str
        Regular expression pattern to match.
    full_match : bool
        If True, the entire output must match. If False, a partial match suffices.
    """

    required_fields = ("input", "actual_output")

    def __init__(
        self,
        pattern: str,
        *,
        full_match: bool = False,
        flags: int = 0,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="RegexMatch", threshold=threshold, **kwargs)
        self.pattern = pattern
        self.full_match = full_match
        self._compiled = re.compile(pattern, flags)

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = test_case.actual_output or ""

        if self.full_match:
            match = self._compiled.fullmatch(actual) is not None
        else:
            match = self._compiled.search(actual) is not None

        return self._result(
            score=1.0 if match else 0.0,
            reason=f"Pattern {'matched' if match else 'not matched'}: {self.pattern}",
        )


class JsonValid(Metric):
    """Passes if ``actual_output`` is valid JSON.

    Optionally checks for required keys.

    Parameters
    ----------
    required_keys : list[str] | None
        JSON keys that must be present in the parsed output.
    """

    required_fields = ("input", "actual_output")

    def __init__(
        self,
        *,
        required_keys: list[str] | None = None,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="JsonValid", threshold=threshold, **kwargs)
        self.required_keys = required_keys or []

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = test_case.actual_output or ""

        try:
            data = json.loads(actual)
        except json.JSONDecodeError as e:
            return self._result(score=0.0, reason=f"Invalid JSON: {e}")

        if self.required_keys and isinstance(data, dict):
            missing = [k for k in self.required_keys if k not in data]
            if missing:
                score = 1.0 - (len(missing) / len(self.required_keys))
                return self._result(
                    score=score,
                    reason=f"Valid JSON but missing keys: {missing}",
                )

        return self._result(score=1.0, reason="Valid JSON.")


class LengthConstraint(Metric):
    """Passes if ``actual_output`` length is within bounds.

    Parameters
    ----------
    min_length : int
        Minimum character length (inclusive).
    max_length : int | None
        Maximum character length (inclusive). None means no upper bound.
    unit : str
        ``"chars"`` or ``"words"``.
    """

    required_fields = ("input", "actual_output")

    def __init__(
        self,
        *,
        min_length: int = 0,
        max_length: int | None = None,
        unit: str = "chars",
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="LengthConstraint", threshold=threshold, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = test_case.actual_output or ""

        if self.unit == "words":
            length = len(actual.split())
        else:
            length = len(actual)

        in_bounds = length >= self.min_length
        if self.max_length is not None:
            in_bounds = in_bounds and length <= self.max_length

        bounds_str = f"[{self.min_length}, {self.max_length or '∞'}] {self.unit}"
        return self._result(
            score=1.0 if in_bounds else 0.0,
            reason=f"Length {length} {self.unit} — {'within' if in_bounds else 'outside'} bounds {bounds_str}.",
        )


class NotEmpty(Metric):
    """Passes if ``actual_output`` is not empty or whitespace-only."""

    required_fields = ("input", "actual_output")

    def __init__(self, *, threshold: float = 1.0, **kwargs: Any) -> None:
        super().__init__(name="NotEmpty", threshold=threshold, **kwargs)

    async def score(self, test_case: TestCase) -> MetricResult:
        actual = (test_case.actual_output or "").strip()
        is_empty = not actual
        return self._result(
            score=0.0 if is_empty else 1.0,
            reason="Output is empty." if is_empty else "Output is non-empty.",
        )


class LatencyCheck(Metric):
    """Passes if ``latency_ms`` on the test case is below a threshold.

    Parameters
    ----------
    max_latency_ms : float
        Maximum acceptable latency in milliseconds.
    """

    required_fields = ("input",)

    def __init__(
        self,
        max_latency_ms: float = 5000.0,
        *,
        threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="LatencyCheck", threshold=threshold, **kwargs)
        self.max_latency_ms = max_latency_ms

    async def score(self, test_case: TestCase) -> MetricResult:
        if test_case.latency_ms is None:
            return self._result(
                score=1.0,
                reason="Latency not measured — auto-pass.",
                metadata={"latency_ms": None},
            )

        within = test_case.latency_ms <= self.max_latency_ms
        # Proportional score: at max_latency → 0.5, at 2x → 0.0
        if within:
            score = 1.0
        else:
            overshoot = test_case.latency_ms / self.max_latency_ms
            score = max(0.0, 1.0 - (overshoot - 1.0))

        return self._result(
            score=score,
            reason=f"Latency {test_case.latency_ms:.0f}ms vs max {self.max_latency_ms:.0f}ms.",
            metadata={"latency_ms": test_case.latency_ms},
        )
