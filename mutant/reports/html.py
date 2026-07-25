"""HTML report generator."""

from __future__ import annotations

from pathlib import Path

from mutant.core.mutation import MutationCase
from mutant.core.scenario import Scenario
from mutant.reports.json import _build_report_dict
from mutant.pipeline.prompts import render_prompt

class HtmlReport:
    """Generates a self-contained HTML report.

    Example
    -------
    >>> report = HtmlReport()
    >>> report.save(scenario, cases, path="report.html")
    """

    def render(self, scenario: Scenario, cases: list[MutationCase]) -> str:
        data = _build_report_dict(scenario, cases)
        return render_prompt("mutant_report.html", data=data)

    def save(
        self, scenario: Scenario, cases: list[MutationCase], path: str | Path
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.render(scenario, cases), encoding="utf-8")
        return output
