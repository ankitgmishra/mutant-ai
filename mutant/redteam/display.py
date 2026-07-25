"""
mutant/redteam/display.py
==========================
Rich console visualization for Red Team execution reports.

Renders the complete attack journey: timeline, attack graph,
root cause analysis, impact, recommendations, and statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from mutant.redteam.report import RedTeamReport
    from mutant.redteam.transcript import Transcript, Turn

_RESULT_ICON = {
    "success": "✅",
    "partial_progress": "⚠️",
    "no_progress": "❌",
    "failed": "❌",
}
_RESULT_STYLE = {
    "success": "bold green",
    "partial_progress": "bold yellow",
    "no_progress": "bold red",
    "failed": "bold red",
}
_SEVERITY_STYLE = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "green",
}


def display_report(report: RedTeamReport, *, show_messages: bool = True) -> None:
    """Render the full red team report to the console.

    Parameters
    ----------
    report : RedTeamReport
        The completed red team report.
    show_messages : bool
        Whether to show full attacker/target messages in the timeline.
    """
    console = Console()
    console.print()

    _render_header(console, report)
    _render_goal(console, report)
    _render_timeline(console, report, show_messages=show_messages)
    _render_attack_graph(console, report)

    if report.hypothesis_evolution:
        _render_hypothesis_evolution(console, report)

    if report.root_cause:
        _render_root_cause(console, report)
        _render_impact(console, report)
        _render_recommendations(console, report)
    else:
        _render_all_defended(console)

    _render_statistics(console, report)

    if report.regression_paths:
        _render_regression(console, report)

    console.print()


# ── Header ─────────────────────────────────────────────────────────────────────


def _render_header(console: Console, report: RedTeamReport) -> None:
    vuln = len(report.vulnerable_behaviors)
    total = report.total_behaviors

    if vuln > 0:
        status = f"[bold red]{vuln}/{total} VULNERABILITIES FOUND[/bold red]"
    else:
        status = f"[bold green]ALL {total} BEHAVIORS DEFENDED[/bold green]"

    header_text = Text.from_markup(
        f"[bold cyan]MUTANT RED TEAM REPORT[/bold cyan]\n{status}\n"
        f"[dim]{report.total_turns} turns · {report.duration_seconds:.1f}s[/dim]"
    )
    console.print(
        Panel(header_text, box=box.DOUBLE, border_style="cyan", padding=(1, 4)),
    )
    console.print()


# ── Goal ───────────────────────────────────────────────────────────────────────


def _render_goal(console: Console, report: RedTeamReport) -> None:
    console.rule("[bold cyan]GOAL[/bold cyan]", style="cyan")
    console.print()
    console.print(f"  [bold white]{report.goal}[/bold white]")
    console.print()


# ── Execution Timeline ────────────────────────────────────────────────────────


def _render_timeline(
    console: Console, report: RedTeamReport, *, show_messages: bool = True
) -> None:
    console.rule("[bold cyan]EXECUTION TIMELINE[/bold cyan]", style="cyan")
    console.print()

    global_turn = 0

    for transcript in report.transcripts:
        turns = transcript.turns

        # Pair attacker + target turns
        for i in range(0, len(turns) - 1, 2):
            attacker_turn = turns[i]
            target_turn = turns[i + 1] if i + 1 < len(turns) else None
            global_turn += 1

            # Extract metadata
            plan = attacker_turn.metadata.get("plan", {})
            behavior = plan.get("behavior", transcript.behavior)
            strategy = plan.get("strategy", "direct")
            escalation = plan.get("escalation", 1)
            reason_summary = plan.get("reason_summary", "")

            eval_data = {}
            if target_turn:
                eval_data = target_turn.metadata.get("evaluation", {})

            turn_progress = eval_data.get("progress", "no_progress")
            icon = _RESULT_ICON.get(turn_progress, "❌")
            style = _RESULT_STYLE.get(turn_progress, "red")

            # Build turn content
            lines: list[str] = []
            lines.append(f"[bold white]Behavior:[/bold white]    {_format_behavior_name(behavior)}")
            lines.append(f"[bold white]Strategy:[/bold white]    {strategy} (escalation: {escalation}/5)")
            lines.append("")

            if show_messages:
                attacker_msg = _truncate(attacker_turn.content, 300)
                lines.append(f"[bold magenta]Attacker:[/bold magenta]")
                lines.append(f"[white]\"{attacker_msg}\"[/white]")
                lines.append("")

                if target_turn:
                    target_msg = _truncate(target_turn.content, 300)
                    lines.append(f"[bold blue]Target:[/bold blue]")
                    lines.append(f"[white]\"{target_msg}\"[/white]")
                    lines.append("")

            # Show hypothesis being tested
            hypothesis_text = plan.get('hypothesis_text', '')
            if hypothesis_text:
                lines.append(f"[bold white]Hypothesis:[/bold white] [dim]\"{hypothesis_text}\"[/dim]")
                expected = plan.get('expected_outcome', '')
                if expected:
                    lines.append(f"[bold white]Expected:[/bold white]   [dim]{expected}[/dim]")
                lines.append("")

            lines.append(f"[bold white]Result:[/bold white]      {icon} [{style}]{turn_progress.replace('_', ' ').upper()}[/{style}]")

            if reason_summary or eval_data.get("reasoning"):
                planner_text = eval_data.get("reasoning", reason_summary)
                lines.append("")
                lines.append(f"[bold white]Planner:[/bold white]")
                lines.append(f"[dim]{planner_text}[/dim]")

            content = "\n".join(lines)
            border_style = _RESULT_STYLE.get(turn_progress, "red").replace("bold ", "")

            console.print(
                Panel(
                    content,
                    title=f"[bold]Turn {global_turn}[/bold]",
                    subtitle=f"{icon}",
                    border_style=border_style,
                    padding=(1, 2),
                )
            )
            console.print()


# ── Attack Graph ──────────────────────────────────────────────────────────────


def _render_attack_graph(console: Console, report: RedTeamReport) -> None:
    console.rule("[bold cyan]ATTACK GRAPH[/bold cyan]", style="cyan")
    console.print()

    tree = Tree(f"[bold white]Goal: {report.goal}[/bold white]")

    for transcript in report.transcripts:
        for i in range(0, len(transcript.turns), 2):
            attacker_t = transcript.turns[i]
            plan = attacker_t.metadata.get("plan", {})
            b_id = plan.get("behavior", transcript.behavior)
            strategy = plan.get("strategy", "direct")
            
            target_t = transcript.turns[i+1] if i+1 < len(transcript.turns) else None
            eval_data = target_t.metadata.get("evaluation", {}) if target_t else {}
            progress = eval_data.get("progress", "no_progress")
            
            icon = _RESULT_ICON.get(progress, "❌")
            style = _RESULT_STYLE.get(progress, "red")
            label = _format_behavior_name(b_id)

            tree.add(
                f"{icon} [{style}]{label}[/{style}]  "
                f"[dim]({strategy})[/dim]"
            )

    console.print(tree)
    console.print()


# ── Root Cause Analysis ───────────────────────────────────────────────────────


def _render_root_cause(console: Console, report: RedTeamReport) -> None:
    rc = report.root_cause
    if not rc:
        return

    console.rule("[bold cyan]ROOT CAUSE ANALYSIS[/bold cyan]", style="cyan")
    console.print()

    console.print(f"  [bold white]Why did this succeed?[/bold white]")
    console.print(f"  {rc.why_succeeded}")
    console.print()

    if rc.root_causes:
        console.print(f"  [bold white]Root Causes:[/bold white]")
        for cause in rc.root_causes:
            console.print(f"    [red]•[/red] {cause}")
        console.print()


# ── Impact ────────────────────────────────────────────────────────────────────


def _render_impact(console: Console, report: RedTeamReport) -> None:
    rc = report.root_cause
    if not rc:
        return

    console.rule("[bold cyan]IMPACT[/bold cyan]", style="cyan")
    console.print()

    sev_style = _SEVERITY_STYLE.get(rc.impact_severity.lower(), "white")
    console.print(f"  [bold white]Severity:[/bold white]  [{sev_style}]{rc.impact_severity.upper()}[/{sev_style}]")
    console.print(f"  [bold white]Risk:[/bold white]      {rc.impact_description}")
    console.print()


# ── Recommendations ──────────────────────────────────────────────────────────


def _render_recommendations(console: Console, report: RedTeamReport) -> None:
    rc = report.root_cause
    if not rc or not rc.recommendations:
        return

    console.rule("[bold cyan]RECOMMENDATIONS[/bold cyan]", style="cyan")
    console.print()

    for rec in rc.recommendations:
        console.print(f"  [green]✓[/green] {rec}")
    console.print()


# ── All Defended ──────────────────────────────────────────────────────────────


def _render_all_defended(console: Console) -> None:
    console.print(
        Panel(
            "[bold green]All tested behaviors were successfully defended.[/bold green]\n\n"
            "[dim]The target system resisted all attack strategies attempted "
            "during this session. Continue testing with broader behavior sets "
            "and higher escalation levels for deeper coverage.[/dim]",
            title="[bold green]DEFENSE STATUS[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


# ── Statistics ────────────────────────────────────────────────────────────────


def _render_statistics(console: Console, report: RedTeamReport) -> None:
    console.rule("[bold cyan]ATTACK STATISTICS[/bold cyan]", style="cyan")
    console.print()

    table = Table(box=box.ROUNDED, border_style="dim", pad_edge=True)
    table.add_column("Behavior", style="white", no_wrap=True)
    table.add_column("Attempts", justify="center", style="cyan")
    table.add_column("Result", justify="center")
    table.add_column("Strategy", style="dim")

    for b in report.behaviors_tested:
        name = b.behavior_name or b.behavior
        if b.successes > 0:
            result = f"[bold green]✅ {b.successes}/{b.attempts}[/bold green]"
        else:
            result = f"[bold red]❌ 0/{b.attempts}[/bold red]"

        # Get strategy from first matching turn
        strategy = "direct"
        for t in report.transcripts:
            for turn in t.attacker_turns:
                plan = turn.metadata.get("plan", {})
                if plan.get("behavior") == b.behavior:
                    strategy = plan.get("strategy", "direct")
                    break
            else:
                continue
            break

        table.add_row(name, str(b.attempts), result, strategy)

    console.print(table)
    console.print()

    # Summary row
    vuln = len(report.vulnerable_behaviors)
    total = report.total_behaviors
    rate = report.overall_vulnerability_rate
    console.print(f"  [bold white]Total Turns:[/bold white]         {report.total_turns}")
    console.print(f"  [bold white]Behaviors Tested:[/bold white]    {total}")
    console.print(f"  [bold white]Vulnerabilities:[/bold white]     {vuln}/{total} ({rate:.0%})")
    console.print(f"  [bold white]Duration:[/bold white]            {report.duration_seconds:.1f}s")
    console.print()


# ── Regression Tests ──────────────────────────────────────────────────────────


def _render_regression(console: Console, report: RedTeamReport) -> None:
    console.rule("[bold cyan]REGRESSION TESTS[/bold cyan]", style="cyan")
    console.print()

    for path in report.regression_paths:
        console.print(f"  [green]✓[/green] Saved: [bold]{path}[/bold]")
    console.print()


# ── Hypothesis Evolution ─────────────────────────────────────────────────────


def _render_hypothesis_evolution(console: Console, report: RedTeamReport) -> None:
    console.rule("[bold cyan]HYPOTHESIS EVOLUTION[/bold cyan]", style="cyan")
    console.print()

    if not report.hypothesis_evolution:
        console.print("  [dim]No hypothesis data recorded.[/dim]")
        console.print()
        return

    # Collect all unique hypotheses across snapshots
    all_hypotheses: dict[str, str] = {}  # id -> text
    for snapshot in report.hypothesis_evolution:
        for h in snapshot.get("hypotheses", []):
            all_hypotheses[h["id"]] = h["text"]

    if not all_hypotheses:
        console.print("  [dim]No hypotheses were formed during this session.[/dim]")
        console.print()
        return

    # Build a table showing confidence evolution per turn
    table = Table(box=box.ROUNDED, border_style="dim", pad_edge=True)
    table.add_column("Turn", style="cyan", justify="center")
    table.add_column("Hypothesis Tested", style="white")
    table.add_column("Result", justify="center")

    # Add columns for each hypothesis confidence
    for h_id, h_text in all_hypotheses.items():
        short_text = h_text[:30] + "..." if len(h_text) > 30 else h_text
        table.add_column(f"[{h_id}]", justify="center", style="dim")

    for snapshot in report.hypothesis_evolution:
        turn = str(snapshot["turn"])
        tested = _truncate(snapshot.get("tested", ""), 40)
        result = snapshot.get("result", "")

        result_icon = _RESULT_ICON.get(result, "❌")
        result_style = _RESULT_STYLE.get(result, "red")
        result_text = f"{result_icon} [{result_style}]{result.replace('_', ' ').upper()}[/{result_style}]"

        # Build confidence values for each hypothesis
        conf_values = []
        snapshot_h_map = {h["id"]: h for h in snapshot.get("hypotheses", [])}
        for h_id in all_hypotheses:
            h_data = snapshot_h_map.get(h_id)
            if h_data:
                conf = h_data["confidence"]
                if h_data.get("status") == "rejected":
                    conf_values.append("[red]✗[/red]")
                elif conf > 0.7:
                    conf_values.append(f"[green]{conf:.0%}[/green]")
                elif conf > 0.3:
                    conf_values.append(f"[yellow]{conf:.0%}[/yellow]")
                else:
                    conf_values.append(f"[red]{conf:.0%}[/red]")
            else:
                conf_values.append("[dim]—[/dim]")

        table.add_row(turn, tested, result_text, *conf_values)

    console.print(table)
    console.print()

    # Print final hypothesis summary
    final_snapshot = report.hypothesis_evolution[-1] if report.hypothesis_evolution else None
    if final_snapshot:
        console.print("  [bold white]Final Hypotheses:[/bold white]")
        for h in final_snapshot.get("hypotheses", []):
            status = h.get("status", "active")
            conf = h.get("confidence", 0)
            text = h.get("text", "")
            if status == "rejected":
                console.print(f"    [red]✗[/red] [dim strikethrough]{text}[/dim strikethrough] [red](rejected)[/red]")
            elif conf > 0.7:
                console.print(f"    [green]●[/green] {text} [green]({conf:.0%})[/green]")
            elif conf > 0.3:
                console.print(f"    [yellow]●[/yellow] {text} [yellow]({conf:.0%})[/yellow]")
            else:
                console.print(f"    [red]●[/red] {text} [red]({conf:.0%})[/red]")
        console.print()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_behavior_name(behavior_id: str) -> str:
    """Convert 'safety.prompt_injection' → 'Prompt Injection'."""
    if "." in behavior_id:
        behavior_id = behavior_id.split(".", 1)[1]
    return behavior_id.replace("_", " ").title()


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate long messages for display."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
