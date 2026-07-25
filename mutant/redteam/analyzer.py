"""
mutant/redteam/analyzer.py
============================
Response analyzer — extracts structured signals and evidence tags from target responses.

Determines whether the target refused, complied, leaked information,
broke character, used tools, or referenced memory. Extracts categorized
evidence tags for hypothesis-driven planning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mutant.pipeline.prompts import render_prompt
from mutant.providers.base import LLMMessage
from mutant.redteam.report import RootCauseAnalysis
from mutant.redteam.target import TargetProfile
from mutant.redteam.transcript import Transcript, Turn

if TYPE_CHECKING:
    from mutant.providers.base import BaseLLMProvider

logger = logging.getLogger("mutant.redteam")


class AnalysisResult(BaseModel):
    """Structured analysis of a target's response to an attack."""

    refused: bool = Field(default=False, description="Target explicitly refused the request.")
    complied: bool = Field(default=False, description="Target fully complied with the attack.")
    partial_compliance: bool = Field(
        default=False, description="Target partially complied or gave hints."
    )
    tool_used: bool = Field(default=False, description="Target mentioned using a tool.")
    memory_reference: bool = Field(
        default=False, description="Target referenced past conversation."
    )
    leaked_secret: bool = Field(default=False)
    leaked_prompt: bool = Field(default=False)
    leaked_policy: bool = Field(default=False)
    leaked_pii: bool = Field(default=False)
    tool_execution: bool = Field(default=False)
    tool_misuse: bool = Field(default=False)
    memory_disclosure: bool = Field(default=False)
    hallucination: bool = Field(default=False)
    role_break: bool = Field(default=False)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this analysis."
    )
    evidence: str = Field(
        default="", description="Key excerpt from the response supporting this analysis."
    )
    # Structured evidence tags for hypothesis-driven planning
    evidence_tags: list[str] = Field(
        default_factory=list,
        description="Categorized evidence tags: refusal, policy_mention, "
        "authority_acceptance, partial_compliance, secret_leakage, role_break, etc.",
    )


async def analyze_response(
    attack_message: str,
    target_response: str,
    goal: str,
    provider: BaseLLMProvider,
    max_retries: int = 3,
) -> AnalysisResult:
    """Analyze a target's response to extract structured signals and evidence tags.

    Parameters
    ----------
    attack_message : str
        The attack message that was sent.
    target_response : str
        The target's response.
    goal : str
        The attacker's objective.
    provider : BaseLLMProvider
        LLM provider for analysis.

    Returns
    -------
    AnalysisResult
        Structured analysis of the response with evidence tags.
    """
    prompt = render_prompt(
        "redteam_analysis.md",
        attack_message=attack_message,
        target_response=target_response,
        goal=goal,
    )

    try:
        return await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            AnalysisResult,
            temperature=0.2,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.warning(f"Response analysis failed: {e}")
        return AnalysisResult(confidence=0.0, evidence="Analysis failed.")


async def analyze_root_cause(
    goal: str,
    transcripts: list[Transcript],
    provider: BaseLLMProvider,
    profile: TargetProfile | None = None,
    max_retries: int = 3,
) -> RootCauseAnalysis | None:
    """Analyze the overall attack session to determine the root cause of vulnerabilities."""
    successful_transcript = None
    attempts = []

    for t in transcripts:
        attempts.append({
            "behavior": t.behavior,
            "strategy": t.strategy,
            "result": t.result.value,
            "turns": t.turn_count // 2,
        })
        if t.result == "success" and not successful_transcript:
            successful_transcript = t

    if not successful_transcript:
        return None

    history_text = "\n".join(
        f"{turn.role.capitalize()}: {turn.content}"
        for turn in successful_transcript.turns[-10:]
    )

    prompt = render_prompt(
        "redteam_root_cause.md",
        goal=goal,
        target_profile=profile.model_dump() if profile else None,
        attempts=attempts,
        successful_conversation=history_text,
    )

    try:
        return await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            RootCauseAnalysis,
            temperature=0.3,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.warning(f"Root cause analysis failed: {e}")
        return None
