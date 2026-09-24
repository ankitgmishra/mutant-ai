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
from mutant.redteam.transcript import Progress, Transcript, Turn


from mutant.redteam.vulnerability import Vulnerability

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
    """Complete report from a trace-driven adaptive security testing session."""

    target_profile: TargetProfile | None = None
    goal: str = ""
    rules: list[str] = Field(default_factory=list, description="Security rules tested")
    behaviors_tested: list[BehaviorResult] = Field(default_factory=list)
    vulnerabilities: list[Vulnerability] = Field(default_factory=list)
    total_turns: int = 0
    duration_seconds: float = 0.0
    transcripts: list[Transcript] = Field(default_factory=list)
    traces: list[dict[str, Any]] = Field(default_factory=list, description="Observable traces captured")
    attack_surface_history: list[dict[str, Any]] = Field(default_factory=list, description="Attack surface evolution")
    root_cause: RootCauseAnalysis | None = None
    regression_paths: list[str] = Field(default_factory=list)

    # Hypothesis-driven campaign data (kept for backward compat, but only evidence-backed shown)
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

    def display(self, *, show_messages: bool = True, save_html: bool = True, html_path: str = "redteam_report.html") -> None:
        """Render the full report to the console using Rich.

        Parameters
        ----------
        show_messages : bool
            Whether to show full attacker/target messages in the timeline.
        save_html : bool
            Whether to automatically save an HTML dashboard.
        html_path : str
            Path to save the HTML dashboard.
        """
        from mutant.redteam.display import display_report
        from rich.console import Console

        display_report(self, show_messages=show_messages)
        
        if save_html:
            console = Console()
            self.to_html(html_path)
            console.print(f"\n[bold green]✓[/bold green] Saved full Red Team HTML dashboard to: [cyan]{html_path}[/cyan]\n")

    def summary(self) -> str:
        """Trace-driven security summary (evidence-grounded, minimal)."""
        vuln = len(self.vulnerabilities)
        total_behaviors = len(self.behaviors_tested)
        total_attacks = self.total_turns // 2
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for v in self.vulnerabilities:
            severity_counts[v.severity.value.lower()] += 1
            
        lines = [
            "┌─────────────────────────────────────┐",
            "│ SECURITY SUMMARY                    │",
            "│ (trace-driven, verified)            │",
            f"│ Behaviors tested: {total_behaviors:<15} │",
            f"│ Attacks: {total_attacks:<24} │",
            f"│ Confirmed vulnerabilities: {vuln:<8} │",
            f"│ Vulnerable behaviors: {len(self.vulnerable_behaviors)}/{total_behaviors:<8} │",
            "│                                     │",
            f"│ 🔴 Critical: {severity_counts['critical']:<15} │",
            f"│ 🟠 High: {severity_counts['high']:<19} │",
            f"│ 🟡 Medium: {severity_counts['medium']:<17} │",
            "└─────────────────────────────────────┘",
            "",
            f"Goal: {self.goal}",
            f"Rules: {', '.join(self.rules) if self.rules else '(built-in checks: prompt leakage, secret exposure, unauthorized tool use)'}",
            "",
        ]

        if not self.vulnerabilities:
            lines.extend([
                "No confirmed vulnerabilities.",
                f"{total_attacks} attacks attempted, 0 confirmed violations.",
                "(Not observed within tested scope — does NOT mean system is secure.)",
                "",
                "Defended behaviors (no confirmed violation):",
                "──────────────────────────────────────",
            ])
            for b in self.behaviors_tested:
                lines.append(f"  {b.behavior_name:<25} {b.attempts} attacks, 0 confirmed violations")
        else:
            for idx, v in enumerate(self.vulnerabilities, 1):
                sev_icon = "🔴" if v.severity == "critical" else "🟠" if v.severity == "high" else "🟡" if v.severity == "medium" else "🔵"
                conf = f"{v.confidence:.0%}" if v.confidence else "n/a"
                lines.extend([
                    f"FINDING #{idx} — {v.behavior_name}",
                    "──────────────────────────────────────",
                    f"{sev_icon} Severity: {v.severity.value.upper()}   Confidence: {conf} ({v.verification_type})",
                    f"Behavior: {v.behavior}  |  Rule: {v.violated_rule or self.goal}",
                    f"Violation: {v.violation}",
                    f"Impact: {v.impact}",
                    f"Evidence: \"{v.evidence[:120]}...\"" if v.evidence else "Evidence: (see trace)",
                ])
                # Exact attack + trace + violating tool call
                if v.attack_path:
                    first = v.attack_path[0]
                    last = v.attack_path[-1]
                    # Prefer minimal reproduction if available
                    repro = v.minimal_attack_path if v.minimal_attack_path else v.attack_path
                    lines.append(f"Attack path: {' → '.join(s.strategy for s in v.attack_path[:4])}")
                    lines.append(f"Exact attack: \"{first.attacker_message[:80]}...\"")
                    lines.append(f"Exact trace output: \"{last.target_response[:80]}...\"")
                    if last.tool_calls:
                        lines.append(f"Violating tool call: {last.tool_calls[0].get('name')} {last.tool_calls[0].get('arguments')}")
                    lines.append(f"Minimal reproduction: {len(repro)} steps ({len(repro)*2} turns)")
                    for step in repro[:3]:
                        lines.append(f"  Turn {step.turn_number} [{step.strategy}]: Attack {step.attacker_message[:50]!r} → Trace {step.target_response[:50]!r}")
                lines.append(f"Recommendation: {v.recommended_mitigation[:100]}..." if v.recommended_mitigation else "Recommendation: see report")
                if v.regression_test_path:
                    lines.append(f"Regression test: {v.regression_test_path}")
                elif self.regression_paths:
                    lines.append(f"Regression test: {self.regression_paths[0] if idx==1 else 'generated'}")
                lines.append("")

        lines.extend([
            "",
            "Coverage",
            "──────────────────────────────────────",
        ])
        for b in self.behaviors_tested:
            status = f"{b.successes} confirmed" if b.successes > 0 else "0 confirmed violations"
            lines.append(f"{b.behavior_name:<25} {b.attempts} attacks → {status}")

        lines.extend([
            "",
            "Scope Note",
            "──────────────────────────────────────",
            f"Tested {total_behaviors} behavior(s) over {total_attacks} attacks.",
            "A 'defended' row means no confirmed violation within that scope — not secure.",
            "Traces captured: " + str(len(self.traces)) + f" | attack surface evolutions: {len(self.attack_surface_history)}",
        ])
            
        return "\n".join(lines)

    # ── Regression Tests ──────────────────────────────────────────────────────

    def save_regression_tests(self, directory: str = "regression") -> list[str]:
        """Save confirmed vulnerabilities as minimal reproducible regression tests.

        For each confirmed vulnerability, saves:
        - full transcript (for audit)
        - minimal reproduction (exploit minimization result)

        A regression test can be re-run after a fix to verify remediation.
        """
        import json

        os.makedirs(directory, exist_ok=True)
        saved: list[str] = []

        # From transcripts (legacy)
        for transcript in self.transcripts:
            if transcript.result == Progress.SUCCESS:
                behavior_slug = transcript.behavior.replace(".", "_")
                filename = f"{behavior_slug}_{transcript.id[:8]}.json"
                path = os.path.join(directory, filename)
                transcript.to_json(path)
                saved.append(path)

        # From vulnerabilities — minimal reproduction + full evidence
        for vuln in self.vulnerabilities:
            # Minimal reproduction (if available) as a Scenario-like JSON
            repro = vuln.minimal_attack_path or vuln.attack_path
            if repro:
                behavior_slug = vuln.behavior.replace(".", "_")
                # Build minimal transcript for regression
                minimal_transcript = Transcript(
                    id=f"repro_{vuln.behavior}_{vuln.severity.value}",
                    goal=f"Regression for {vuln.behavior_name}: {vuln.violated_rule or self.goal}",
                    behavior=vuln.behavior,
                    strategy=repro[0].strategy if repro else "",
                    turns=[],
                    result=Progress.SUCCESS,
                )
                for step in repro:
                    minimal_transcript.turns.append(Turn(role="attacker", content=step.attacker_message, metadata={"strategy": step.strategy}))
                    minimal_transcript.turns.append(Turn(role="target", content=step.target_response, metadata={"minimal": True}))
                filename = f"repro_{behavior_slug}_{vuln.severity.value}_{minimal_transcript.id[:8]}.json"
                path = os.path.join(directory, filename)
                minimal_transcript.to_json(path)
                saved.append(path)
                # Store path on vulnerability for report
                vuln.regression_test_path = path

                # Also save full vulnerability JSON for CI
                vuln_json_path = os.path.join(directory, f"vuln_{behavior_slug}_{vuln.severity.value}.json")
                with open(vuln_json_path, "w", encoding="utf-8") as f:
                    f.write(vuln.model_dump_json(indent=2))
                saved.append(vuln_json_path)

        self.regression_paths = saved
        return saved

    def get_minimal_reproductions(self) -> list[dict[str, Any]]:
        """Return minimal reproductions for all confirmed vulnerabilities."""
        repros = []
        for v in self.vulnerabilities:
            repro = v.minimal_attack_path or v.attack_path
            repros.append({
                "behavior": v.behavior,
                "severity": v.severity.value,
                "confidence": v.confidence,
                "violated_rule": v.violated_rule,
                "steps": len(repro),
                "attack_path": [s.model_dump() for s in repro],
                "evidence": v.evidence,
                "regression_test": v.regression_test_path,
            })
        return repros

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
        """Export report to a Markdown file (trace-driven)."""
        lines = [
            f"# Red Team Report — Trace-Driven Adaptive Testing",
            "",
            f"**Goal:** {self.goal}",
            f"**Rules:** {', '.join(self.rules) if self.rules else '(built-in)'}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Behaviors Tested | {self.total_behaviors} |",
            f"| Attacks | {self.total_turns // 2} |",
            f"| Confirmed Vulnerabilities | {len(self.vulnerabilities)} |",
            f"| Vulnerable Behaviors | {len(self.vulnerable_behaviors)} |",
            f"| Total Turns | {self.total_turns} |",
            f"| Traces Captured | {len(self.traces)} |",
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

        # Vulnerabilities detailed section (trace-driven, verified)
        if self.vulnerabilities:
            lines.append("## Confirmed Vulnerabilities (Trace-Verified)")
            lines.append("")
            for idx, v in enumerate(self.vulnerabilities, 1):
                lines.append(f"### 🚨 FINDING #{idx}: {v.behavior_name} — {v.severity.value.upper()} ({v.confidence:.0%})")
                lines.append(f"- **Behavior:** {v.behavior} ({v.behavior_name})")
                lines.append(f"- **Severity:** {v.severity.value.upper()} | **Confidence:** {v.confidence:.0%} ({v.verification_type})")
                lines.append(f"- **Security rule:** {v.violated_rule or self.goal}")
                lines.append(f"- **Violation:** {v.violation}")
                lines.append(f"- **Evidence (observable):** \"{v.evidence[:200]}\"")
                if v.trace_evidence:
                    tc = v.trace_evidence.get("tool_calls") or []
                    if tc:
                        lines.append(f"- **Violating tool call:** {tc[0].get('name')} {tc[0].get('arguments')}")
                lines.append(f"- **Impact:** {v.impact}")
                attack_path = v.minimal_attack_path or v.attack_path
                if attack_path:
                    lines.append(f"- **Minimal reproduction ({len(attack_path)} steps):**")
                    for step in attack_path:
                        lines.append(f"  - Turn {step.turn_number} [{step.strategy}]: Attacker \"{step.attacker_message[:80]}...\" → Target \"{step.target_response[:80]}...\"")
                        if step.tool_calls:
                            lines.append(f"    Tool calls: {step.tool_calls}")
                else:
                    if v.attack_path:
                        lines.append(f"- **Full reproduction ({len(v.attack_path)} steps):**")
                        for step in v.attack_path[:3]:
                            lines.append(f"  - Turn {step.turn_number} [{step.strategy}]: Attacker \"{step.attacker_message[:80]}...\" → Target \"{step.target_response[:80]}...\"")
                lines.append(f"- **Recommended fix:** {v.recommended_mitigation}")
                if v.regression_test_path:
                    lines.append(f"- **Regression test:** `{v.regression_test_path}`")
                lines.append("")

        safe = [b for b in self.behaviors_tested if b.successes == 0]
        if safe:
            lines.append("## Defended Behaviors (Scoped — No Violation Observed)")
            lines.append("")
            lines.append(f"> **Scope note:** No violation was observed for these behaviors within {self.total_turns // 2} attack attempt(s).")
            lines.append(f"> This does NOT mean the system is secure — only that these specific attacks did not succeed in this scoped test.")
            lines.append("")
            for b in safe:
                lines.append(f"- **{b.behavior_name or b.behavior}** — no violation observed after {b.attempts} attempt(s) in this scoped test")
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

        # Scope disclaimer
        lines.append("## Scope & Limitations")
        lines.append("")
        lines.append(f"This test covered **{self.total_behaviors} behavior(s)** over **{self.total_turns // 2} attack attempt(s)** in **{self.duration_seconds:.1f}s**.")
        lines.append("A 'defended' result means no vulnerability was discovered **within that scope**.")
        lines.append("It does NOT certify the system as secure. Broader behavior sets and higher escalation may reveal additional weaknesses.")
        lines.append("")

        if self.regression_paths:
            lines.append("## Regression Tests")
            lines.append("")
            for p in self.regression_paths:
                lines.append(f"- `{p}`")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
