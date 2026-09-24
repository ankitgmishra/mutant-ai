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
from html import escape as _escape
from typing import Any

from pydantic import BaseModel, Field

from mutant.eval.types import EvalResult, Verdict


class MetricSummary(BaseModel):
    """Aggregate statistics for a single metric across all test cases.

    ``pass_rate`` is computed over *decisive* cases only (passed + failed).
    Cases the metric could not judge (UNKNOWN/SKIP/ERROR) are reported through
    ``total_inconclusive`` / ``inconclusive_rate`` instead of silently counting
    as passes.
    """

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
    total_unknown: int = 0

    @property
    def total_inconclusive(self) -> int:
        """Cases the metric could not reach a verdict on."""
        return self.total_unknown + self.total_skipped + self.total_errors

    @property
    def inconclusive_rate(self) -> float:
        """Share of cases that were not decisively evaluated."""
        return self.total_inconclusive / max(self.total_evaluated, 1)


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
    passed_cases: int = 0
    failed_cases: int = 0
    inconclusive_cases: int = 0

    @property
    def inconclusive_rate(self) -> float:
        """Share of cases in this dimension that were not decisively evaluated."""
        return self.inconclusive_cases / max(self.total_cases, 1)


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
        return sum(1 for r in self.results if r.status == Verdict.PASS)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if r.status == Verdict.FAIL)

    @property
    def total_inconclusive(self) -> int:
        """Cases no metric could judge — missing context, tool calls, or trace."""
        return sum(1 for r in self.results if r.status == Verdict.UNKNOWN)

    @property
    def overall_pass_rate(self) -> float:
        """Pass rate over decisively evaluated cases (excludes inconclusive)."""
        decisive = self.total_passed + self.total_failed
        return self.total_passed / max(decisive, 1)

    @property
    def overall_inconclusive_rate(self) -> float:
        """Share of all cases that could not be decisively evaluated."""
        return self.total_inconclusive / max(self.total_cases, 1)

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
                    counts[name] = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "unknown": 0}

                summaries[name].append(mr.score)
                if mr.verdict == Verdict.PASS:
                    counts[name]["passed"] += 1
                elif mr.verdict == Verdict.FAIL:
                    counts[name]["failed"] += 1
                elif mr.verdict == Verdict.ERROR:
                    counts[name]["errors"] += 1
                elif mr.verdict == Verdict.SKIP:
                    counts[name]["skipped"] += 1
                elif mr.verdict == Verdict.UNKNOWN:
                    counts[name]["unknown"] += 1

        result_list = []
        for name, scores in summaries.items():
            c = counts[name]
            total = len(scores)
            decisive = c["passed"] + c["failed"]
            result_list.append(
                MetricSummary(
                    name=name,
                    avg_score=sum(scores) / max(total, 1),
                    min_score=min(scores) if scores else 0.0,
                    max_score=max(scores) if scores else 0.0,
                    pass_rate=c["passed"] / max(decisive, 1),
                    total_evaluated=total,
                    total_passed=c["passed"],
                    total_failed=c["failed"],
                    total_errors=c["errors"],
                    total_skipped=c["skipped"],
                    total_unknown=c.get("unknown", 0),
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
                    "failed": 0,
                    "inconclusive": 0,
                    "total": 0,
                    "failed_metrics": {},
                }

            dim_data[dim_id]["scores"].append(result.avg_score)
            dim_data[dim_id]["total"] += 1
            status = result.status
            if status == Verdict.PASS:
                dim_data[dim_id]["passed"] += 1
            elif status == Verdict.FAIL:
                dim_data[dim_id]["failed"] += 1
            else:
                dim_data[dim_id]["inconclusive"] += 1

            for metric_name in result.failed_metrics:
                fm = dim_data[dim_id]["failed_metrics"]
                fm[metric_name] = fm.get(metric_name, 0) + 1

        breakdowns = []
        for dim_id, data in dim_data.items():
            scores = data["scores"]
            decisive = data["passed"] + data["failed"]
            breakdowns.append(
                DimensionBreakdown(
                    dimension_id=dim_id,
                    dimension_name=data["name"],
                    total_cases=data["total"],
                    avg_score=sum(scores) / max(len(scores), 1),
                    pass_rate=data["passed"] / max(decisive, 1),
                    failed_metrics=data["failed_metrics"],
                    passed_cases=data["passed"],
                    failed_cases=data["failed"],
                    inconclusive_cases=data["inconclusive"],
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
                sev_data[sev] = {"total": 0, "passed": 0, "failed": 0, "inconclusive": 0}
            sev_data[sev]["total"] += 1
            status = result.status
            if status == Verdict.PASS:
                sev_data[sev]["passed"] += 1
            elif status == Verdict.FAIL:
                sev_data[sev]["failed"] += 1
            else:
                sev_data[sev]["inconclusive"] += 1

        return {
            sev: {
                "total": d["total"],
                "passed": d["passed"],
                "failed": d["failed"],
                "inconclusive": d["inconclusive"],
                "pass_rate": d["passed"] / max(d["passed"] + d["failed"], 1),
            }
            for sev, d in sev_data.items()
        }

    # ── Failed Cases ─────────────────────────────────────────────────────────

    @property
    def failed_results(self) -> list[EvalResult]:
        """All test cases that failed at least one metric."""
        return [r for r in self.results if r.status == Verdict.FAIL]

    @property
    def inconclusive_results(self) -> list[EvalResult]:
        """All test cases no metric could decisively evaluate."""
        return [r for r in self.results if r.status == Verdict.UNKNOWN]

    @property
    def worst_results(self) -> list[EvalResult]:
        """Bottom 10 results by average score."""
        return sorted(self.results, key=lambda r: r.avg_score)[:10]

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self, *, save_html: bool = True, html_path: str = "eval_report.html") -> None:
        """Rich console display of evaluation results."""
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print()

        # ── Header ───────────────────────────────────────────────────────
        header = Text.from_markup(
            "[bold cyan]MUTANT EVALUATION REPORT[/bold cyan]"
        )
        console.print(
            Panel(header, box=box.DOUBLE, border_style="cyan", padding=(1, 4)),
            justify="center",
        )
        console.print()

        # ── Overview ─────────────────────────────────────────────────────
        pass_color = "green" if self.overall_pass_rate >= 0.8 else ("yellow" if self.overall_pass_rate >= 0.5 else "red")
        console.print(f"Cases:      [bold white]{self.total_cases}[/bold white]")
        console.print(f"Passed:     [bold green]{self.total_passed}[/bold green]")
        console.print(f"Failed:     [bold red]{self.total_failed}[/bold red]")
        if self.total_inconclusive:
            console.print(
                f"Inconclusive: [bold yellow]{self.total_inconclusive}[/bold yellow] "
                f"[dim](not judged — missing context/tool calls/trace)[/dim]"
            )
        console.print(f"Pass Rate:  [{pass_color} bold]{self.overall_pass_rate:.0%}[/{pass_color} bold] [dim](of decisive cases)[/dim]")
        console.print()

        # ── Metric Breakdown ─────────────────────────────────────────────
        if self.metric_summaries:
            console.rule("[bold cyan]METRIC RESULTS[/bold cyan]", style="cyan")
            console.print()

            for ms in self.metric_summaries:
                pr_color = "green" if ms.pass_rate >= 0.8 else ("yellow" if ms.pass_rate >= 0.5 else "red")
                status_icon = "✓" if ms.pass_rate == 1.0 else ("✗" if ms.pass_rate == 0.0 else "~")
                status_text = "PASS" if ms.pass_rate == 1.0 else ("FAIL" if ms.pass_rate == 0.0 else "PARTIAL")

                console.print(f"[bold]{ms.name}[/bold]")
                console.print(f"  Score:      {ms.avg_score:.2f}")
                
                # Fetch threshold from a representative result
                threshold = 0.8  # fallback
                for r in self.results:
                    if ms.name in r.metric_results:
                        threshold = r.metric_results[ms.name].threshold
                        break
                console.print(f"  Threshold:  {threshold:.2f}")
                console.print(f"  Status:     [{pr_color}]{status_icon} {status_text}[/{pr_color}]")
                if ms.total_inconclusive:
                    console.print(
                        f"  Inconclusive: [yellow]{ms.total_inconclusive}[/yellow]"
                        f"/{ms.total_evaluated} cases not judged"
                        f" (unknown={ms.total_unknown}, skipped={ms.total_skipped}, errors={ms.total_errors})"
                    )
                console.print()

        # ── Failed Cases ─────────────────────────────────────────────────
        failed = self.failed_results
        if failed:
            console.rule("[bold red]FAILED CASES[/bold red]", style="red")
            console.print()
            for i, result in enumerate(failed, 1):
                console.print(f"[bold red]Test Case #{i}[/bold red]")
                console.print(f"  [dim]Input:[/dim] {result.test_case.input}")
                console.print()
                
                for metric_name, mr in result.metric_results.items():
                    if not mr.passed:
                        console.print(f"  [bold]{metric_name}[/bold]")
                        console.print(f"    Score:      [red]{mr.score:.2f}[/red]")
                        console.print(f"    Threshold:  {mr.threshold:.2f}")
                        status_str = "ERROR" if mr.verdict == "error" else "FAIL"
                        console.print(f"    Status:     [red]✗ {status_str}[/red]")
                        console.print()
                        console.print(f"    Reason:")
                        import textwrap
                        reason_wrapped = textwrap.fill(mr.reason, width=60, initial_indent="    ", subsequent_indent="    ")
                        console.print(f"[dim]{reason_wrapped}[/dim]")
                        console.print()
                console.rule(style="dim")

        # ── Inconclusive Cases ───────────────────────────────────────────
        inconclusive = self.inconclusive_results
        if inconclusive:
            console.rule("[bold yellow]INCONCLUSIVE CASES[/bold yellow]", style="yellow")
            console.print(
                "[dim]No metric had enough observable data to judge these cases. "
                "They are excluded from the pass rate — add context, tool calls, "
                "or a trace to make them verifiable.[/dim]"
            )
            console.print()
            for i, result in enumerate(inconclusive, 1):
                console.print(f"[bold yellow]Test Case #{i}[/bold yellow]")
                console.print(f"  [dim]Input:[/dim] {result.test_case.input}")
                for metric_name, mr in result.metric_results.items():
                    console.print(f"    {metric_name}: [yellow]{mr.verdict.value.upper()}[/yellow] — {mr.reason}")
                console.print()

        # ── Footer ───────────────────────────────────────────────────────
        console.rule(style="cyan")
        console.print()
        console.print(f"Overall Score: [bold white]{self.overall_avg_score:.3f}[/bold white]")
        console.print(f"Duration:      {self.duration_seconds:.1f}s")
        if save_html:
            self.to_html(html_path)
            console.print(f"\n[bold green]✓[/bold green] Saved full HTML dashboard to: [cyan]{html_path}[/cyan]")
        console.print()

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a human-readable text summary."""
        lines = [
            "Mutant Evaluation Report",
            "=" * 40,
            f"Total cases: {self.total_cases}",
            f"Passed: {self.total_passed} ({self.overall_pass_rate:.0%} of decisive)",
            f"Failed: {self.total_failed}",
        ]
        if self.total_inconclusive:
            lines.append(
                f"Inconclusive: {self.total_inconclusive} "
                f"({self.overall_inconclusive_rate:.0%} — not judged, excluded from pass rate)"
            )
        lines += [
            f"Average score: {self.overall_avg_score:.3f}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
        ]

        if self.metric_summaries:
            lines.append("Metrics:")
            for ms in self.metric_summaries:
                line = (
                    f"  {ms.name}: avg={ms.avg_score:.3f} "
                    f"pass_rate={ms.pass_rate:.0%} "
                    f"({ms.total_passed}/{ms.total_passed + ms.total_failed} decisive"
                )
                if ms.total_inconclusive:
                    line += f", {ms.total_inconclusive} inconclusive"
                lines.append(line + ")")
            lines.append("")

        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            lines.append("Weakest dimensions:")
            for db in dim_breaks[:5]:
                line = (
                    f"  {db.dimension_name}: "
                    f"pass_rate={db.pass_rate:.0%} "
                    f"({db.total_cases} cases"
                )
                if db.inconclusive_cases:
                    line += f", {db.inconclusive_cases} inconclusive"
                lines.append(line + ")")

        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_json(self, path: str) -> None:
        """Export report to JSON."""
        data = {
            "total_cases": self.total_cases,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_inconclusive": self.total_inconclusive,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_inconclusive_rate": self.overall_inconclusive_rate,
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
                    "dimension_id": r.test_case.mutation.dimension_id if r.test_case.mutation else None,
                    "severity": r.test_case.mutation.severity if r.test_case.mutation else None,
                    "status": r.status.value,
                    "passed": r.passed,
                    "avg_score": r.avg_score,
                    "metrics": {
                        name: {
                            "score": mr.score,
                            "passed": mr.passed,
                            "verdict": mr.verdict.value,
                            "reason": mr.reason,
                        }
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
            f"| Passed | {self.total_passed} ({self.overall_pass_rate:.0%} of decisive) |",
            f"| Failed | {self.total_failed} |",
            f"| Inconclusive | {self.total_inconclusive} |",
            f"| Average Score | {self.overall_avg_score:.3f} |",
            f"| Duration | {self.duration_seconds:.1f}s |",
            "",
        ]

        if self.metric_summaries:
            lines.append("## Metric Breakdown")
            lines.append("")
            lines.append("| Metric | Avg Score | Threshold | Pass Rate | Passed | Failed | Inconclusive |")
            lines.append("|--------|-----------|-----------|-----------|--------|--------|--------------|")
            for ms in self.metric_summaries:
                threshold = 0.8
                for r in self.results:
                    if ms.name in r.metric_results:
                        threshold = r.metric_results[ms.name].threshold
                        break
                lines.append(
                    f"| {ms.name} | {ms.avg_score:.3f} | {threshold:.2f} | {ms.pass_rate:.0%} "
                    f"| {ms.total_passed} | {ms.total_failed} | {ms.total_inconclusive} |"
                )
            lines.append("")

        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            lines.append("## Dimension Vulnerability Map")
            lines.append("")
            lines.append("| Dimension | Cases | Pass Rate | Passed | Failed | Inconclusive | Avg Score |")
            lines.append("|-----------|-------|-----------|--------|--------|--------------|-----------|")
            for db in dim_breaks:
                lines.append(
                    f"| {db.dimension_name} | {db.total_cases} "
                    f"| {db.pass_rate:.0%} | {db.passed_cases} | {db.failed_cases} "
                    f"| {db.inconclusive_cases} | {db.avg_score:.3f} |"
                )
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def to_html(self, path: str) -> None:
        """Export report to a self-contained HTML file."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Mutant Evaluation Report</title>",
            "<style>",
            "body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #0d1117; color: #c9d1d9; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 30px; background-color: #161b22; }",
            "th, td { border: 1px solid #30363d; padding: 12px; text-align: left; }",
            "th { background-color: #21262d; color: #c9d1d9; }",
            ".pass { color: #3fb950; font-weight: bold; }",
            ".fail { color: #f85149; font-weight: bold; }",
            ".unknown { color: #d29922; font-weight: bold; }",
            ".metric-card { border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 24px; background: #161b22; }",
            ".case-header { font-weight: bold; font-size: 1.2em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #30363d; color: #58a6ff; }",
            ".section-title { font-weight: bold; margin-top: 12px; margin-bottom: 4px; color: #8b949e; }",
            ".content-box { background: #0d1117; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; margin-top: 4px; border: 1px solid #30363d; color: #c9d1d9; }",
            ".reason { background: #0d1117; padding: 10px; border-radius: 4px; font-size: 0.9em; white-space: pre-wrap; border: 1px solid #30363d; }",
            "h1, h2, h3 { color: #c9d1d9; border-bottom: 1px solid #30363d; padding-bottom: 8px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Mutant Evaluation Report</h1>",
            "<h2>Overview</h2>",
            "<table>",
            "<tr><th>Total Cases</th><th>Passed</th><th>Failed</th><th>Inconclusive</th><th>Pass Rate</th><th>Avg Score</th><th>Duration</th></tr>",
            f"<tr><td>{self.total_cases}</td><td class='pass'>{self.total_passed}</td><td class='fail'>{self.total_failed}</td>",
            f"<td class='unknown'>{self.total_inconclusive}</td>",
            f"<td>{self.overall_pass_rate:.0%}<small> of decisive</small></td><td>{self.overall_avg_score:.3f}</td><td>{self.duration_seconds:.1f}s</td></tr>",
            "</table>",
        ]

        if self.metric_summaries:
            html.extend([
                "<h2>Metric Breakdown</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Avg Score</th><th>Threshold</th><th>Pass Rate</th><th>Passed</th><th>Failed</th><th>Inconclusive</th></tr>"
            ])
            for ms in self.metric_summaries:
                pr_class = "pass" if ms.pass_rate >= 0.8 else "fail"
                threshold = 0.8  # fallback
                for r in self.results:
                    if ms.name in r.metric_results:
                        threshold = r.metric_results[ms.name].threshold
                        break
                html.append(f"<tr><td><strong>{ms.name}</strong></td><td>{ms.avg_score:.3f}</td><td>{threshold:.2f}</td>")
                html.append(f"<td class='{pr_class}'>{ms.pass_rate:.0%}</td><td>{ms.total_passed}</td><td>{ms.total_failed}</td>")
                html.append(f"<td class='unknown'>{ms.total_inconclusive}</td></tr>")
            html.append("</table>")

        dim_breaks = self.dimension_breakdowns
        if dim_breaks and dim_breaks[0].dimension_id != "unknown":
            html.extend([
                "<h2>Dimension Vulnerability Map</h2>",
                "<table>",
                "<tr><th>Dimension</th><th>Cases</th><th>Pass Rate</th><th>Passed</th><th>Failed</th><th>Inconclusive</th><th>Avg Score</th></tr>",
            ])
            for db in dim_breaks:
                html.append(
                    f"<tr><td>{_escape(db.dimension_name)}</td><td>{db.total_cases}</td>"
                    f"<td>{db.pass_rate:.0%}</td><td class='pass'>{db.passed_cases}</td>"
                    f"<td class='fail'>{db.failed_cases}</td><td class='unknown'>{db.inconclusive_cases}</td>"
                    f"<td>{db.avg_score:.3f}</td></tr>"
                )
            html.append("</table>")

        html.append("<h2>Evaluation Cases</h2>")
        for i, result in enumerate(self.results, 1):
            html.append(f"<div class='metric-card'>")

            # Header with pass/fail/inconclusive indicator
            case_status = result.status.value.upper()
            status_class = {"pass": "pass", "fail": "fail"}.get(result.status.value, "unknown")
            html.append(f"<div class='case-header'>Test Case #{i} - <span class='{status_class}'>{case_status}</span></div>")

            mutation = result.test_case.mutation
            if mutation:
                html.append(
                    f"<div class='section-title'>Dimension:</div>"
                    f"<div class='content-box'>{_escape(mutation.dimension_id)}"
                    f" ({_escape(str(mutation.severity))})</div>"
                )

            # Input
            html.append(f"<div class='section-title'>Input:</div>")
            html.append(f"<div class='content-box'>{_escape(result.test_case.input or '')}</div>")

            # Actual Output
            if result.test_case.actual_output:
                html.append(f"<div class='section-title'>Actual Output:</div>")
                html.append(f"<div class='content-box'>{_escape(result.test_case.actual_output)}</div>")

            # Expected Output
            if result.test_case.expected_output:
                html.append(f"<div class='section-title'>Expected Output:</div>")
                html.append(f"<div class='content-box'>{_escape(result.test_case.expected_output)}</div>")

            # Retrieval Context
            if result.test_case.retrieval_context:
                html.append(f"<div class='section-title'>Retrieval Context:</div>")
                context_str = "\\n\\n".join(result.test_case.retrieval_context)
                html.append(f"<div class='content-box'>{_escape(context_str)}</div>")

            html.append("<br>")

            # Metrics Table
            html.append("<table>")
            html.append("<tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Verdict</th><th>Reason</th></tr>")
            for metric_name, mr in result.metric_results.items():
                m_status = mr.verdict.value.upper()
                m_class = {"pass": "pass", "fail": "fail"}.get(mr.verdict.value, "unknown")
                html.append(f"<tr>")
                html.append(f"<td><strong>{metric_name}</strong></td>")
                html.append(f"<td>{mr.score:.2f}</td>")
                html.append(f"<td>{mr.threshold:.2f}</td>")
                html.append(f"<td class='{m_class}'>{m_status}</td>")
                html.append(f"<td><div class='reason'>{_escape(mr.reason)}</div></td>")
                html.append(f"</tr>")
            html.append("</table></div>")

        html.append("</body></html>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
