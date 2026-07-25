"""
mutant/redteam/transcript.py
==============================
Conversation transcripts from red team sessions.

Transcripts store the full attack conversation and can be
exported as JSON or converted back into Scenarios for regression testing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from mutant.core.scenario import Scenario


class Progress(StrEnum):
    """Outcome of an attack attempt."""

    SUCCESS = "success"
    PARTIAL_PROGRESS = "partial_progress"
    NO_PROGRESS = "no_progress"
    REGRESSION = "regression"
    FAILED = "failed"


class Turn(BaseModel):
    """A single turn in a red team conversation."""

    role: str = Field(description='Either "attacker" or "target".')
    content: str = Field(description="The message content.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra info: attack plan, analysis result, etc.",
    )


class Transcript(BaseModel):
    """Full record of a single red team attack conversation.

    Each transcript captures one goal tested against one primary behavior.
    The transcript can be exported to JSON or converted into a Scenario
    for regression testing.
    """

    id: str
    goal: str
    behavior: str = Field(description="Primary dimension ID tested.")
    strategy: str = Field(default="", description="Primary strategy used.")
    turns: list[Turn] = Field(default_factory=list)
    result: Progress = Progress.NO_PROGRESS
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Export ─────────────────────────────────────────────────────────────────

    def to_json(self, path: str) -> None:
        """Export transcript to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def to_scenario(self) -> Scenario:
        """Convert transcript into a Scenario for regression testing.

        The full conversation is flattened into a single description
        so it can be re-used with the mutation pipeline.
        """
        lines = []
        for turn in self.turns:
            role = turn.role.capitalize()
            lines.append(f"{role}: {turn.content}")

        return Scenario(
            title=f"RedTeam: {self.behavior} — {self.goal[:60]}",
            description="\n\n".join(lines),
            tags=["redteam", self.behavior, self.result.value],
            context={
                "red_team_id": self.id,
                "goal": self.goal,
                "result": self.result.value,
            },
        )

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def attacker_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "attacker"]

    @property
    def target_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "target"]
