"""
mutant/eval/report.py
======================
EvalReport — aggregated evaluation results with display and export.

Provides:
  - Per-metric aggregate scores and pass rates.
  - Per-dimension breakdown (Mutant-specific: which mutation dimensions
    caused the most failures).
  - Per-severity breakdown.
  - Rich console display.
  - JSON / Markdown export.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from mutant.eval.types import EvalResult, Verdict


class MetricSummary(BaseModel):
    """Aggregate statistics for a single metric across all test cases."""

    name: str
    avg_score: float
    min_score: float
    max_score: float
    pass_rate: float
    total_evaluated: int
    total_passed: int
    total_failed: int
    total_errors: int
    total_skipped: int


class DimensionBreakdown(BaseModel):
    """Evaluation breakdown for a single mutation dimension."""

    dimension_id: str
    dimension_name: str
    total_cases: int
    avg_score: float
    pass_rate: float
    failed_metrics: dict[str, int] = Field(
        default_factory=dict,
        description="Metric name → count of failures.",
    )


class EvalReport(BaseModel):
    """Complete evaluation report.

    Aggregates all individual EvalResult objects into summary statistics,
    metric breakdowns, dimension breakdowns, and severity analysis.

    Parameters
    ----------
    results : list[EvalResult]
        Individual test case results.
    metrics_used : list[str]
        Names of metrics that were applied.
    duration_seconds : float
        Total wall-clock evaluation time.
    """

    results: list[EvalResult] = Field(default_factory=list)
    metrics_used: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

    # ── Core Properties ──────────────────────────────────────────────────────

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def overall_pass_rate(self) -> float:
        return self.total_passed / max(self.total_cases, 1)

    @property
    def overall_avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.avg_score for r in self.results) / len(self.results)

    # ── Metric Summaries ─────────────────────────────────────────────────────

    @property
    def metric_summaries(self) -> list[MetricSummary]:
        """Per-metric aggregate statistics."""
        summaries: dict[str, list[float]] = {}
        counts: dict[str, dict[str, int]] = {}

        for result in self.results:
            for name, mr in result.metric_results.items():
                if name not in summaries:
                    summaries[name] = []
                    counts[name] = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

                summaries[name].append(mr.score)
                if mr.verdict == Verdict.PASS:
                    counts[name]["passed"] += 1
                elif mr.verdict == Verdict.FAIL:
                    counts[name]["failed"] += 1
                elif mr.verdict == Verdict.ERROR:
                    counts[name]["errors"] += 1
                elif mr.verdict == Verdict.SKIP:
                    counts[name]["skipped"] += 1

        result_list = []
        for name, scores in summaries.items():
            c = counts[name]
            total = len(scores)
            result_list.append(
                MetricSummary(
                    name=name,
                    avg_score=sum(scores) / max(total, 1),
                    min_score=min(scores) if scores else 0.0,
                    max_score=max(scores) if scores else 0.0,
                    pass_rate=c["passed"] / max(total, 1),
                    total_evaluated=total,
                    total_passed=c["passed"],
                    total_failed=c["failed"],
                    total_errors=c["errors"],
                    total_skipped=c["skipped"],
                )
            )
        return result_list

    # ── Dimension Breakdowns (Mutant-specific) ───────────────────────────────

    @property
    def dimension_breakdowns(self) -> list[DimensionBreakdown]:
        """Per-dimension evaluation breakdown.

        Groups results by the mutation dimension that generated the test case.
        Shows which behavioral dimensions cause the most failures.
        """
        dim_data: dict[str, dict[str, Any]] = {}

        for result in self.results:
            mutation = result.test_case.mutation
            dim_id = (mutation.dimension_id if mutation else None) or "unknown"
            dim_name = (mutation.dimension_name if mutation else None) or dim_id

            if dim_id not in dim_data:
                dim_data[dim_id] = {
                    "name": dim_name,
                    "scores": [],
                    "passed": 0,
                    "total": 0,
                    "failed_metrics": {},
                }

            dim_data[dim_id]["scores"].append(result.avg_score)
            dim_data[dim_id]["total"] += 1
            if result.passed:
                dim_data[dim_id]["passed"] += 1

            for metric_name in result.failed_metrics:
                fm = dim_data[dim_id]["failed_metrics"]
                fm[metric_name] = fm.get(metric_name, 0) + 1

        breakdowns = []
        for dim_id, data in dim_data.items():
            scores = data["scores"]
            breakdowns.append(
                DimensionBreakdown(
                    dimension_id=dim_id,
                    dimension_name=data["name"],
                    total_cases=data["total"],
                    avg_score=sum(scores) / max(len(scores), 1),
                    pass_rate=data["passed"] / max(data["total"], 1),
                    failed_metrics=data["failed_metrics"],
                )
            )

        # Sort by pass_rate ascending — worst dimensions first
        breakdowns.sort(key=lambda b: b.pass_rate)
        return breakdowns

    # ── Severity Breakdown ───────────────────────────────────────────────────

    @property
    def severity_breakdown(self) -> dict[str, dict[str, Any]]:
        """Pass rates grouped by mutation severity."""
        sev_data: dict[str, dict[str, int]] = {}

        for result in self.results:
            mutation = result.test_case.mutation
            sev = (mutation.severity if mutation else None) or "unknown"
            if sev not in sev_data:
                sev_data[sev] = {"total": 0, "passed": 0}
            sev_data[sev]["total"] += 1
            if result.passed:
                sev_data[sev]["passed"] += 1

        return {
            sev: {
                "total": d["total"],
                "passed": d["passed"],
                "failed": d["total"] - d["passed"],
                "pass_rate": d["passed"] / max(d["total"], 1),
            }
            for sev, d in sev_data.items()
        }

    # ── Failed Cases ─────────────────────────────────────────────────────────

    @property
    def failed_results(self) -> list[EvalResult]:
        """All test cases that failed at least one metric."""
        return [r for r in self.results if not r.passed]

    @property
    def worst_results(self) -> list[EvalResult]:
        """Bottom 10 results by average score."""
        return sorted(self.results, key=lambda r: r.avg_score)[:10]

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self) -> None:
        """Rich console display of evaluation results."""
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        console.print()

        # ── Header ───────────────────────────────────────────────────────
        header = Text.from_markup(
            "[bold cyan]MUTANT EVALUATION REPORT[/bold cyan]\n"
            "[dim]Mutation-Driven Quality Assessment[/dim]"
        )
        console.print(
            Panel(header, box=box.DOUBLE, border_style="cyan", padding=(1, 4)),
            justify="center",
        )
        console.print()

        # ── Overview ─────────────────────────────────────────────────────
        pass_color = "green" if self.overall_pass_rate >= 0.8 else (
            "yellow" if self.overall_pass_rate >= 0.5 else "red"
        )
        console.print(f"  [dim]Total Cases:[/dim]    [bold white]{self.total_cases}[/bold white]")
        console.print(f"  [dim]Passed:[/dim]          [bold green]{self.total_passed}[/bold green]")
        console.print(f"  [dim]Failed:[/dim]          [bold red]{self.total_failed}[/bold red]")
        console.print(f"  [dim]Pass Rate:[/dim]       [{pass_color} bold]{self.overall_pass_rate:.0%}[/{pass_color} bold]")
        console.print(f"  [dim]Avg Score:[/dim]       [bold white]{self.overall_avg_score:.3f}[/bold white]")
        console.print(f"  [dim]Duration:[/dim]        [white]{self.duration_seconds:.1f}s[/white]")
        console.print()

        # ── Metric Breakdown ─────────────────────────────────────────────
        if self.metric_summaries:
            console.rule("[bold cyan]METRIC BREAKDOWN[/bold cyan]", style="cyan")
            console.print()

            table = Table(box=box.SIMPLE_HEAD, show_edge=False)
            table.add_column("Metric", style="bold white")
            table.add_column("Avg Score", justify="center")
            table.add_column("Pass Rate", justify="center")
            table.add_column("Passed", justify="center", style="green")
            table.add_column("Failed", justify="center", style="red")
            table.add_column("Errors", justify="center", style="yellow")

            for ms in self.metric_summaries:
                pr_color = "green" if ms.pass_rate >= 0.8 else (
                    "yellow" if ms.pass_rate >= 0.5 else "red"
                )
                table.add_row(
                    ms.name,
                    f"{ms.avg_score:.3f}",
                    f"[{pr_color}]{ms.pass_rate:.0%}[/{pr_color}]",
                    str(ms.total_passed),
                    str(ms.total_failed),
                    str(ms.total_errors),
                )
            console.print(table)
            console.print()

        # ── Dimension Breakdown (Mutant-specific) ────────────────────────
        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            console.rule("[bold cyan]DIMENSION VULNERABILITY MAP[/bold cyan]", style="cyan")
            console.print()

            table = Table(box=box.SIMPLE_HEAD, show_edge=False)
            table.add_column("Dimension", style="bold white")
            table.add_column("Cases", justify="center")
            table.add_column("Avg Score", justify="center")
            table.add_column("Pass Rate", justify="center")
            table.add_column("Top Failing Metric", justify="center")

            for db in dim_breaks:
                pr_color = "green" if db.pass_rate >= 0.8 else (
                    "yellow" if db.pass_rate >= 0.5 else "red"
                )
                top_fail = ""
                if db.failed_metrics:
                    top_fail = max(db.failed_metrics, key=db.failed_metrics.get)
                    top_fail = f"[red]{top_fail} ({db.failed_metrics[top_fail]})[/red]"

                table.add_row(
                    db.dimension_name,
                    str(db.total_cases),
                    f"{db.avg_score:.3f}",
                    f"[{pr_color}]{db.pass_rate:.0%}[/{pr_color}]",
                    top_fail,
                )
            console.print(table)
            console.print()

        # ── Severity Breakdown ───────────────────────────────────────────
        sev = self.severity_breakdown
        if sev and "unknown" not in sev:
            console.rule("[bold cyan]SEVERITY IMPACT[/bold cyan]", style="cyan")
            console.print()

            # Display in severity order
            severity_order = ["critical", "high", "medium", "low"]
            for s in severity_order:
                if s in sev:
                    d = sev[s]
                    pr_color = "green" if d["pass_rate"] >= 0.8 else (
                        "yellow" if d["pass_rate"] >= 0.5 else "red"
                    )
                    sev_color = {"critical": "red bold", "high": "red", "medium": "yellow", "low": "green"}.get(s, "white")
                    console.print(
                        f"  [{sev_color}]{s.upper():>10}[/{sev_color}]  "
                        f"[{pr_color}]{d['pass_rate']:.0%}[/{pr_color}] pass rate  "
                        f"({d['passed']}/{d['total']} passed)"
                    )
            console.print()

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a human-readable text summary."""
        lines = [
            "Mutant Evaluation Report",
            "=" * 40,
            f"Total cases: {self.total_cases}",
            f"Passed: {self.total_passed} ({self.overall_pass_rate:.0%})",
            f"Failed: {self.total_failed}",
            f"Average score: {self.overall_avg_score:.3f}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
        ]

        if self.metric_summaries:
            lines.append("Metrics:")
            for ms in self.metric_summaries:
                lines.append(
                    f"  {ms.name}: avg={ms.avg_score:.3f} "
                    f"pass_rate={ms.pass_rate:.0%} "
                    f"({ms.total_passed}/{ms.total_evaluated})"
                )
            lines.append("")

        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            lines.append("Weakest dimensions:")
            for db in dim_breaks[:5]:
                lines.append(
                    f"  {db.dimension_name}: "
                    f"pass_rate={db.pass_rate:.0%} "
                    f"({db.total_cases} cases)"
                )

        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_json(self, path: str) -> None:
        """Export report to JSON."""
        data = {
            "total_cases": self.total_cases,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_avg_score": self.overall_avg_score,
            "duration_seconds": self.duration_seconds,
            "metrics": [ms.model_dump() for ms in self.metric_summaries],
            "dimensions": [db.model_dump() for db in self.dimension_breakdowns],
            "severity": self.severity_breakdown,
            "results": [
                {
                    "input": r.test_case.input[:200],
                    "actual_output": (r.test_case.actual_output or "")[:200],
                    "dimension": r.test_case.mutation.dimension_name if r.test_case.mutation else None,
                    "severity": r.test_case.mutation.severity if r.test_case.mutation else None,
                    "passed": r.passed,
                    "avg_score": r.avg_score,
                    "metrics": {
                        name: {"score": mr.score, "passed": mr.passed, "reason": mr.reason}
                        for name, mr in r.metric_results.items()
                    },
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def to_markdown(self, path: str) -> None:
        """Export report to Markdown."""
        lines = [
            "# Mutant Evaluation Report",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Cases | {self.total_cases} |",
            f"| Passed | {self.total_passed} ({self.overall_pass_rate:.0%}) |",
            f"| Failed | {self.total_failed} |",
            f"| Average Score | {self.overall_avg_score:.3f} |",
            f"| Duration | {self.duration_seconds:.1f}s |",
            "",
        ]

        if self.metric_summaries:
            lines.append("## Metric Breakdown")
            lines.append("")
            lines.append("| Metric | Avg Score | Pass Rate | Passed | Failed |")
            lines.append("|--------|-----------|-----------|--------|--------|")
            for ms in self.metric_summaries:
                lines.append(
                    f"| {ms.name} | {ms.avg_score:.3f} | {ms.pass_rate:.0%} "
                    f"| {ms.total_passed} | {ms.total_failed} |"
                )
            lines.append("")

        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            lines.append("## Dimension Vulnerability Map")
            lines.append("")
            lines.append("| Dimension | Cases | Pass Rate | Avg Score |")
            lines.append("|-----------|-------|-----------|-----------|")
            for db in dim_breaks:
                lines.append(
                    f"| {db.dimension_name} | {db.total_cases} "
                    f"| {db.pass_rate:.0%} | {db.avg_score:.3f} |"
                )
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
