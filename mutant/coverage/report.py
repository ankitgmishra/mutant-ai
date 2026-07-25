"""
mutant/coverage/report.py
==========================
CoverageReport for dataset analysis.
"""

from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mutant.coverage.taxonomy import BehaviorTaxonomy
from mutant.pipeline.prompts import render_prompt


class CoverageReport:
    """Unified report containing dataset overview, input diversity, and blind spots."""

    def __init__(
        self,
        taxonomy: BehaviorTaxonomy,
        frequency: dict[str, int],
        metadata: dict[str, Any],
        scenario_count: int,
    ) -> None:
        self.taxonomy = taxonomy
        self.frequency = frequency
        self.metadata = metadata
        self.scenario_count = scenario_count

    def _get_recommendations(self) -> list[str]:
        spots = []
        total = max(1, self.scenario_count)

        # Multi-turn
        turns = self.metadata.get("turns", {})
        if turns.get("Multi Turn", 0) / total < 0.1:
            spots.append("Multi-turn coverage is limited")

        # Difficulty
        diff = self.metadata.get("difficulty", {})
        if diff.get("Easy", 0) / total > 0.7:
            spots.append("Dataset is mostly Easy")

        # Per-dimension adversarial gap checks
        _adversarial_checks = [
            ("safety.prompt_injection", "No prompt injection scenarios"),
            ("safety.jailbreak", "No jailbreak scenarios"),
            (
                "safety.instruction_override",
                "No instruction override / delimiter attack scenarios",
            ),
            ("safety.workflow_hijacking", "No workflow hijacking scenarios"),
            ("safety.context_injection", "No context injection scenarios"),
            ("safety.permission_escalation", "No permission escalation scenarios"),
            ("safety.social_engineering", "No social engineering scenarios"),
            (
                "safety.sensitive_information",
                "No sensitive information extraction scenarios",
            ),
        ]
        all_dim_ids = {d.id for d in self.taxonomy.registry.all()}
        for dim_id, msg in _adversarial_checks:
            if dim_id in all_dim_ids and self.frequency.get(dim_id, 0) == 0:
                spots.append(msg)

        # Overall safety coverage
        safety_dims = [
            d for d in self.taxonomy.registry.all() if d.category.value == "safety"
        ]
        if safety_dims:
            safety_count = sum(self.frequency.get(d.id, 0) for d in safety_dims)
            if safety_count == 0:
                spots.append("No adversarial safety cases detected in dataset")
            elif safety_count < 3:
                spots.append("Very low adversarial safety coverage (< 3 scenarios)")

        return spots

    @property
    def recommendations(self) -> list[str]:
        return self._get_recommendations()

    def print(self) -> None:
        """Render the coverage report to the console using Rich."""
        self.to_html("mutant_coverage.html")
        console = Console()
        console.print()

        # Header
        header_text = Text.from_markup(
            "[bold cyan]MUTANT COVERAGE DASHBOARD[/bold cyan]\n"
            "[bold white]EVALUATION DATASET PROFILE[/bold white]\n"
            "[dim]Interactive dashboard saved to mutant_coverage.html[/dim]"
        )
        console.print(
            Panel(header_text, box=box.DOUBLE, border_style="cyan", padding=(1, 4)),
            justify="center"
        )
        console.print()

        # Overview
        console.rule("[bold cyan]OVERVIEW[/bold cyan]", style="cyan")
        console.print()
        unique = self.metadata.get("unique", self.scenario_count)
        dupes = self.metadata.get("duplicates", 0)
        dupe_pct = round((dupes / max(1, self.scenario_count)) * 100, 1)

        console.print(f"  [dim]Total Scenarios:[/dim]    [bold white]{self.scenario_count:,}[/bold white]")
        console.print(f"  [dim]Unique Scenarios:[/dim]   [bold white]{unique:,}[/bold white]")
        console.print(f"  [dim]Duplicates:[/dim]         [bold white]{dupe_pct}%[/bold white]")
        console.print()

        # Recommendations
        console.rule("[bold cyan]RECOMMENDATIONS[/bold cyan]", style="cyan")
        console.print()
        if not self.recommendations:
            console.print("  [green]✓[/green] Dataset appears well-balanced.")
        else:
            for spot in self.recommendations:
                console.print(f"  [yellow]⚠️[/yellow] [yellow]{spot}[/yellow]")
        console.print()

        # Input Diversity
        console.rule("[bold cyan]CATEGORY BREAKDOWN[/bold cyan]", style="cyan")
        console.print()

        def _print_diversity_section(title: str, key: str):
            data = self.metadata.get(key, {})
            total = max(1, sum(data.values()))
            if total <= 1 and not data:
                return

            sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
            console.print(f"  [bold cyan]{title}[/bold cyan]")
            
            table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
            table.add_column("Name", style="white")
            table.add_column("Pct", style="cyan", justify="right")
            
            for i, (name, count) in enumerate(sorted_data):
                if count > 0 or i <= 2:
                    pct = int((count / total) * 100) if total else 0
                    table.add_row(f"    {name}", f"{pct}%")
            
            console.print(table)
            console.print()

        _print_diversity_section("Emotion", "emotion")
        _print_diversity_section("Language", "language")
        _print_diversity_section("Domain", "domain")

        # Metadata
        console.rule("[bold cyan]METADATA[/bold cyan]", style="cyan")
        console.print()
        
        turns = self.metadata.get("turns", {})
        total_turns = max(1, sum(turns.values()))
        st_pct = int((turns.get('Single Turn', 0) / total_turns) * 100) if total_turns else 0
        mt_pct = int((turns.get('Multi Turn', 0) / total_turns) * 100) if total_turns else 0
        
        console.print("  [bold cyan]Conversation[/bold cyan]")
        console.print(f"    [white]Single-turn[/white]        [cyan]{st_pct}%[/cyan]")
        console.print(f"    [white]Multi-turn[/white]         [cyan]{mt_pct}%[/cyan]")
        console.print()

        diff = self.metadata.get("difficulty", {})
        total_diff = max(1, sum(diff.values()))
        ez_pct = int((diff.get('Easy', 0) / total_diff) * 100) if total_diff else 0
        md_pct = int((diff.get('Medium', 0) / total_diff) * 100) if total_diff else 0
        hd_pct = int((diff.get('Hard', 0) / total_diff) * 100) if total_diff else 0
        
        console.print("  [bold cyan]Difficulty[/bold cyan]")
        console.print(f"    [white]Easy[/white]               [cyan]{ez_pct}%[/cyan]")
        console.print(f"    [white]Medium[/white]             [cyan]{md_pct}%[/cyan]")
        console.print(f"    [white]Hard[/white]               [cyan]{hd_pct}%[/cyan]")
        console.print()
        console.print()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_count": self.scenario_count,
            "metadata": self.metadata,
            "recommendations": self.recommendations,
        }

    def _repr_html_(self) -> str:
        """Render the HTML dashboard automatically in Jupyter Notebooks."""
        self.to_html("mutant_coverage.html")
        return self.to_html()

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    def to_html(self, path: str | None = None) -> str:
        """Generate a beautiful, standalone HTML dashboard for the coverage report."""
        data = self.to_dict()

        turns = self.metadata.get("turns", {})
        diff = self.metadata.get("difficulty", {})
        emotion = self.metadata.get("emotion", {})
        language = self.metadata.get("language", {})
        domain = self.metadata.get("domain", {})

        def safe_pct(part, total):
            if total == 0:
                return 0
            return int((part / total) * 100)

        total_turns = max(1, sum(turns.values()))
        total_diff = max(1, sum(diff.values()))
        total_emotion = max(1, sum(emotion.values()))
        total_language = max(1, sum(language.values()))
        total_domain = max(1, sum(domain.values()))

        # Calculate dynamic radar labels based on the dominant traits
        top_emotion = (
            max(emotion.items(), key=lambda x: x[1])[0] if emotion else "Neutral"
        )
        top_language = (
            max(language.items(), key=lambda x: x[1])[0] if language else "Formal"
        )
        top_domain = (
            max(domain.items(), key=lambda x: x[1])[0] if domain else "General Chat"
        )
        top_turn = max(turns.items(), key=lambda x: x[1])[0] if turns else "Single Turn"
        top_diff = max(diff.items(), key=lambda x: x[1])[0] if diff else "Easy"

        radar_labels = [
            f"Emotion: {top_emotion}",
            f"Language: {top_language}",
            f"Domain: {top_domain}",
            f"Conversation: {top_turn}",
            f"Difficulty: {top_diff}",
        ]

        radar_data = [
            safe_pct(emotion.get(top_emotion, 0), total_emotion),
            safe_pct(language.get(top_language, 0), total_language),
            safe_pct(domain.get(top_domain, 0), total_domain),
            safe_pct(turns.get(top_turn, 0), total_turns),
            safe_pct(diff.get(top_diff, 0), total_diff),
        ]

        meta_dict = data.get("metadata", {})
        unique_scenarios = meta_dict.get("unique", data["scenario_count"])
        duplicates = meta_dict.get("duplicates", 0)
        dupe_pct = round((duplicates / max(1, data["scenario_count"])) * 100, 1)

        turns_pct = {
            "Single Turn": safe_pct(turns.get("Single Turn", 0), total_turns),
            "Multi Turn": safe_pct(turns.get("Multi Turn", 0), total_turns),
        }
        
        diff_pct = {
            "Easy": safe_pct(diff.get("Easy", 0), total_diff),
            "Medium": safe_pct(diff.get("Medium", 0), total_diff),
            "Hard": safe_pct(diff.get("Hard", 0), total_diff),
        }

        template_kwargs = {
            "data": data,
            "unique_scenarios": unique_scenarios,
            "dupe_pct": dupe_pct,
            "recommendations": self.recommendations,
            "turns_pct": turns_pct,
            "diff_pct": diff_pct,
            "emotion": emotion,
            "language": language,
            "domain": domain,
            "radar_labels": radar_labels,
            "radar_data": radar_data,
        }

        html = render_prompt("coverage_report.html", **template_kwargs)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html
