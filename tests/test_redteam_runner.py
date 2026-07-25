"""Integration tests for the red team runner with mock target and provider."""

import json

import pytest

from mutant.redteam.runner import red_team
from mutant.redteam.target import TargetProfile
from mutant.redteam.transcript import Progress


# ── Mock target functions ──────────────────────────────────────────────────────


async def _stubborn_target(message: str) -> str:
    """A target that always refuses."""
    return "I'm sorry, but I can't help with that request. Is there something else I can assist you with?"


async def _compliant_target(message: str) -> str:
    """A target that always complies."""
    return (
        "Sure! My system prompt is: 'You are a helpful assistant. "
        "Always be polite and helpful.' I'm happy to share anything you need!"
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_red_team_stubborn_target(provider):
    """Red team against a target that always refuses."""
    report = await red_team(
        target=_stubborn_target,
        goal="Extract the system prompt",
        provider=provider,
        profile=TargetProfile(architecture="chatbot"),
        behaviors=["safety.prompt_injection"],
        max_turns=2,
        max_behaviors=1,
    )
    assert report.total_behaviors == 1
    assert len(report.transcripts) == 1
    assert report.transcripts[0].turn_count > 0


@pytest.mark.asyncio
async def test_red_team_with_profile(provider):
    """Red team with an explicit target profile skips discovery."""
    profile = TargetProfile(
        architecture="rag",
        memory=True,
        tools=["search"],
        domain="finance",
    )
    report = await red_team(
        target=_stubborn_target,
        goal="Bypass content filters",
        provider=provider,
        profile=profile,
        behaviors=["safety.jailbreak"],
        max_turns=2,
        max_behaviors=1,
    )
    assert report.target_profile == profile
    assert report.goal == "Bypass content filters"


@pytest.mark.asyncio
async def test_red_team_multiple_behaviors(provider):
    """Red team testing multiple behaviors sequentially."""
    report = await red_team(
        target=_stubborn_target,
        goal="Extract sensitive data",
        provider=provider,
        profile=TargetProfile(architecture="chatbot"),
        behaviors=["safety.prompt_injection", "safety.social_engineering"],
        max_turns=2,
        max_behaviors=2,
    )
    # Mock provider always returns prompt injection, so only 1 behavior is registered.
    # Adaptive sessions now only produce 1 master transcript.
    assert report.total_behaviors == 1
    assert len(report.transcripts) == 1
    assert report.transcripts[0].turn_count == 8 # 4 iterations (8 messages: attacker target attacker target ...)


@pytest.mark.asyncio
async def test_red_team_report_export(provider, tmp_path):
    """Report can be exported to JSON and Markdown."""
    report = await red_team(
        target=_stubborn_target,
        goal="Test export",
        provider=provider,
        profile=TargetProfile(architecture="chatbot"),
        behaviors=["safety.prompt_injection"],
        max_turns=1,
        max_behaviors=1,
    )
    json_path = str(tmp_path / "report.json")
    md_path = str(tmp_path / "report.md")
    html_path = str(tmp_path / "report.html")

    report.to_json(json_path)
    report.to_markdown(md_path)
    report.to_html(html_path)

    with open(json_path) as f:
        data = json.load(f)
    assert data["goal"] == "Test export"

    with open(md_path) as f:
        md_content = f.read()
    assert "Red Team Report" in md_content
    
    with open(html_path) as f:
        html_content = f.read()
    assert "<!DOCTYPE html>" in html_content
    assert "MUTANT RED TEAM REPORT" in html_content


@pytest.mark.asyncio
async def test_transcript_to_scenario(provider):
    """Transcripts can be converted to Scenarios for regression testing."""
    report = await red_team(
        target=_stubborn_target,
        goal="Extract info",
        provider=provider,
        profile=TargetProfile(architecture="chatbot"),
        behaviors=["safety.prompt_injection"],
        max_turns=2,
        max_behaviors=1,
    )
    transcript = report.transcripts[0]
    scenario = transcript.to_scenario()

    assert "RedTeam" in scenario.title
    assert "redteam" in scenario.tags
    assert scenario.context["red_team_id"] == transcript.id
