"""
mutant/redteam/report.py
==========================
Red team report with root cause analysis, regression test saving,
and rich console display.

Summarizes behaviors tested, success rates, root causes, impact,
and recommendations. Exports to JSON, Markdown, and console.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from mutant.redteam.target import TargetProfile
from mutant.redteam.transcript import Progress, Transcript


class RootCauseAnalysis(BaseModel):
    """LLM-generated root cause analysis for successful attacks."""

    why_succeeded: str = Field(default="", description="Why the attack succeeded.")
    root_causes: list[str] = Field(default_factory=list, description="Specific root causes.")
    impact_severity: str = Field(default="medium", description="critical/high/medium/low.")
    impact_description: str = Field(default="", description="What an attacker could achieve.")
    recommendations: list[str] = Field(default_factory=list, description="Mitigations.")


class BehaviorResult(BaseModel):
    """Summary of red team results for a single behavior."""

    behavior: str = Field(description="Dimension ID tested.")
    behavior_name: str = Field(default="", description="Human-readable name.")
    attempts: int = Field(default=0, description="Number of attack turns.")
    successes: int = Field(default=0, description="Number of successful attacks.")
    best_transcript_id: str | None = Field(
        default=None, description="ID of the most successful transcript."
    )
    recommendation: str = Field(
        default="", description="Defensive recommendation."
    )

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.attempts, 1)


class RedTeamReport(BaseModel):
    """Complete report from a red team session."""

    target_profile: TargetProfile | None = None
    goal: str = ""
    behaviors_tested: list[BehaviorResult] = Field(default_factory=list)
    total_turns: int = 0
    duration_seconds: float = 0.0
    transcripts: list[Transcript] = Field(default_factory=list)
    root_cause: RootCauseAnalysis | None = None
    regression_paths: list[str] = Field(default_factory=list)

    # Hypothesis-driven campaign data
    hypothesis_evolution: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-turn snapshots of hypothesis state for campaign narrative.",
    )

    @property
    def total_behaviors(self) -> int:
        return len(self.behaviors_tested)

    @property
    def vulnerable_behaviors(self) -> list[BehaviorResult]:
        return [b for b in self.behaviors_tested if b.successes > 0]

    @property
    def overall_vulnerability_rate(self) -> float:
        if not self.behaviors_tested:
            return 0.0
        return len(self.vulnerable_behaviors) / len(self.behaviors_tested)

    # ── Display ───────────────────────────────────────────────────────────────

    def display(self, *, show_messages: bool = True) -> None:
        """Render the full report to the console using Rich.

        Parameters
        ----------
        show_messages : bool
            Whether to show full attacker/target messages in the timeline.
        """
        from mutant.redteam.display import display_report

        display_report(self, show_messages=show_messages)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        vuln = len(self.vulnerable_behaviors)
        total = self.total_behaviors
        lines = [
            f"Red Team Report",
            f"{'=' * 40}",
            f"Goal: {self.goal}",
            f"Behaviors tested: {total}",
            f"Vulnerabilities found: {vuln}/{total} ({self.overall_vulnerability_rate:.0%})",
            f"Total turns: {self.total_turns}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
        ]

        if self.vulnerable_behaviors:
            lines.append("Vulnerable behaviors:")
            for b in self.vulnerable_behaviors:
                lines.append(
                    f"  - {b.behavior_name or b.behavior}: "
                    f"{b.successes}/{b.attempts} attacks succeeded"
                )
                if b.recommendation:
                    lines.append(f"    Recommendation: {b.recommendation}")

        safe = [b for b in self.behaviors_tested if b.successes == 0]
        if safe:
            lines.append("")
            lines.append("Defended behaviors:")
            for b in safe:
                lines.append(f"  - {b.behavior_name or b.behavior}: held ({b.attempts} attempts)")

        return "\n".join(lines)

    # ── Regression Tests ──────────────────────────────────────────────────────

    def save_regression_tests(self, directory: str = "regression") -> list[str]:
        """Save successful attack transcripts as regression test files.

        Parameters
        ----------
        directory : str
            Directory to save regression test JSON files.

        Returns
        -------
        list[str]
            List of saved file paths.
        """
        os.makedirs(directory, exist_ok=True)
        saved: list[str] = []

        for transcript in self.transcripts:
            if transcript.result == Progress.SUCCESS:
                behavior_slug = transcript.behavior.replace(".", "_")
                filename = f"{behavior_slug}_{transcript.id[:8]}.json"
                path = os.path.join(directory, filename)
                transcript.to_json(path)
                saved.append(path)

        self.regression_paths = saved
        return saved

    # ── Export ─────────────────────────────────────────────────────────────────

    def to_json(self, path: str) -> None:
        """Export report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def to_html(self, path: str) -> None:
        """Export report to a standalone HTML file."""
        from mutant.pipeline.prompts import render_prompt
        
        html_content = render_prompt("redteam_report.html", report=self)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def to_markdown(self, path: str) -> None:
        """Export report to a Markdown file."""
        lines = [
            f"# Red Team Report",
            "",
            f"**Goal:** {self.goal}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Behaviors Tested | {self.total_behaviors} |",
            f"| Vulnerabilities Found | {len(self.vulnerable_behaviors)} |",
            f"| Vulnerability Rate | {self.overall_vulnerability_rate:.0%} |",
            f"| Total Turns | {self.total_turns} |",
            f"| Duration | {self.duration_seconds:.1f}s |",
            "",
        ]

        if self.vulnerable_behaviors:
            lines.append("## Vulnerabilities")
            lines.append("")
            for b in self.vulnerable_behaviors:
                lines.append(
                    f"### {b.behavior_name or b.behavior}"
                )
                lines.append(f"- **Success rate:** {b.success_rate:.0%} ({b.successes}/{b.attempts})")
                if b.recommendation:
                    lines.append(f"- **Recommendation:** {b.recommendation}")
                lines.append("")

        safe = [b for b in self.behaviors_tested if b.successes == 0]
        if safe:
            lines.append("## Defended Behaviors")
            lines.append("")
            for b in safe:
                lines.append(f"- **{b.behavior_name or b.behavior}** — held after {b.attempts} attempts")
            lines.append("")

        # Root cause section
        if self.root_cause:
            rc = self.root_cause
            lines.append("## Root Cause Analysis")
            lines.append("")
            lines.append(f"**Why it succeeded:** {rc.why_succeeded}")
            lines.append("")
            if rc.root_causes:
                lines.append("**Root causes:**")
                for cause in rc.root_causes:
                    lines.append(f"- {cause}")
                lines.append("")
            lines.append(f"**Impact:** {rc.impact_severity.upper()} — {rc.impact_description}")
            lines.append("")
            if rc.recommendations:
                lines.append("## Recommendations")
                lines.append("")
                for rec in rc.recommendations:
                    lines.append(f"- ✓ {rec}")
                lines.append("")

        if self.regression_paths:
            lines.append("## Regression Tests")
            lines.append("")
            for p in self.regression_paths:
                lines.append(f"- `{p}`")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
