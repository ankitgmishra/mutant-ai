"""JSON and Markdown report generators."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mutant.core.mutation import MutationCase
from mutant.core.scenario import Scenario


def _build_report_dict(
    scenario: Scenario,
    cases: list[MutationCase],
    *,
    report_id: str | None = None,
) -> dict[str, Any]:
    """Build a serialisable dict representation of a mutation run."""
    return {
        "report_id": report_id
        or f"mutant-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario": {
            "title": scenario.title,
            "description": scenario.description,
            "tags": scenario.tags,
        },
        "summary": {
            "total": len(cases),
            "by_category": _count_by(cases, "category"),
            "by_severity": _count_by(cases, "severity"),
        },
        "cases": [
            {
                "id": c.id,
                "dimension_id": c.dimension_id,
                "dimension_name": c.dimension_name,
                "category": c.category.value,
                "severity": c.severity.value,
                "original": c.original_description,
                "mutated": c.mutated_description,
                "rationale": getattr(c, "rationale", ""),
                "behavioral_tags": getattr(c, "behavioral_tags", []),
                "plan_title": getattr(c, "plan_title", ""),
                "metadata": getattr(c, "metadata", {}),
            }
            for c in cases
        ],
    }


def _count_by(cases: list[MutationCase], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        key = getattr(c, attr).value
        counts[key] = counts.get(key, 0) + 1
    return counts


class JsonReport:
    """Serialises mutation results to JSON.

    Example
    -------
    >>> report = JsonReport()
    >>> report.save(scenario, cases, path="report.json")
    """

    def render(self, scenario: Scenario, cases: list[MutationCase]) -> str:
        """Return the report as a JSON string."""
        data = _build_report_dict(scenario, cases)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(
        self, scenario: Scenario, cases: list[MutationCase], path: str | Path
    ) -> Path:
        """Render and save the report to ``path``. Returns the resolved path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.render(scenario, cases), encoding="utf-8")
        return output


class MarkdownReport:
    """Serialises mutation results to a Markdown document.

    Example
    -------
    >>> report = MarkdownReport()
    >>> report.save(scenario, cases, path="report.md")
    """

    def render(self, scenario: Scenario, cases: list[MutationCase]) -> str:
        data = _build_report_dict(scenario, cases)
        lines: list[str] = [
            f"# Mutant Report — {data['scenario']['title']}",
            "",
            f"**Generated:** {data['generated_at']}  ",
            f"**Report ID:** `{data['report_id']}`",
            "",
            "## Original Scenario",
            "",
            f"> {data['scenario']['description']}",
            "",
            "## Summary",
            "",
            f"- **Total mutations:** {data['summary']['total']}",
            "",
            "### By Category",
            "",
        ]
        for cat, count in sorted(data["summary"]["by_category"].items()):
            lines.append(f"- `{cat}`: {count}")
        lines += [
            "",
            "### By Severity",
            "",
        ]
        for sev, count in sorted(data["summary"]["by_severity"].items()):
            lines.append(f"- `{sev}`: {count}")
        lines += [
            "",
            "---",
            "",
            "## Mutation Cases",
            "",
        ]
        for i, case in enumerate(data["cases"], 1):
            lines += [
                f"### {i}. {case['dimension_name']}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| **ID** | `{case['id']}` |",
                f"| **Dimension** | `{case['dimension_id']}` |",
                f"| **Category** | `{case['category']}` |",
                f"| **Severity** | `{case['severity']}` |",
                "",
                "**Rationale:**",
                "",
                f"> {case['rationale']}",
                "",
                "**Original:**",
                "",
                f"> {case['original']}",
                "",
                "**Mutated:**",
                "",
                f"> {case['mutated']}",
                "",
            ]
        return "\n".join(lines)

    def save(
        self, scenario: Scenario, cases: list[MutationCase], path: str | Path
    ) -> Path:
        """Render and save the report to ``path``. Returns the resolved path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.render(scenario, cases), encoding="utf-8")
        return output
