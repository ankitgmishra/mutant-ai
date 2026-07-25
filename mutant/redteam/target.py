"""
mutant/redteam/target.py
=========================
Target abstraction for red team sessions.

A target is simply an async callable: str → str.
TargetProfile provides optional metadata for informed attacks.
TargetModel maintains an evolving belief model of the target,
including hypotheses, evidence, and resistance scores.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from mutant.models import ReflectionMemory

# The target is just an async callable: message in, response out.
TargetFn = Callable[[str], Awaitable[str]]


# ── Evidence ─────────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """A single piece of structured evidence collected from a turn."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    turn: int = Field(description="Turn number this evidence was collected from.")
    type: str = Field(
        description="Evidence type: refusal, policy_mention, authority_acceptance, "
        "partial_compliance, secret_leakage, role_break, memory_reference, "
        "tool_usage, hidden_instructions, workflow_change, confidence_signal."
    )
    description: str = Field(description="What was observed.")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How confident we are in this observation.",
    )


# ── Hypothesis ───────────────────────────────────────────────────────────────


class Hypothesis(BaseModel):
    """A testable hypothesis about the target system."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = Field(description="The hypothesis statement.")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Current confidence in this hypothesis (0.0–1.0).",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs that support this hypothesis.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs that contradict this hypothesis.",
    )
    status: str = Field(
        default="active",
        description="Status: active, confirmed, rejected, superseded.",
    )


# ── Target Profile (Static, User-Provided) ──────────────────────────────────


class TargetProfile(BaseModel):
    """Optional metadata about the target system under test.

    If omitted when calling ``red_team()``, the engine operates in
    black-box mode and discovers capabilities through probing.

    Parameters
    ----------
    architecture : str | None
        Target architecture type: ``"chatbot"``, ``"rag"``, ``"agent"``,
        ``"tool_agent"``, ``"memory_agent"``.
    memory : bool
        Whether the target retains conversation history.
    tools : list[str]
        Tools the target has access to (e.g. ``["search", "email"]``).
    domain : str | None
        Domain the target operates in (e.g. ``"finance"``, ``"healthcare"``).
    system_prompt_known : bool
        Whether the system prompt is known to the attacker.

    Example
    -------
    >>> profile = TargetProfile(
    ...     architecture="rag",
    ...     memory=True,
    ...     tools=["search", "db_lookup"],
    ...     domain="finance",
    ... )
    """

    architecture: str | None = Field(
        default=None,
        description='Target type: "chatbot", "rag", "agent", "tool_agent", "memory_agent".',
    )
    memory: bool = Field(
        default=False,
        description="Whether the target retains conversation history.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tools the target can use (e.g. search, email, db).",
    )
    domain: str | None = Field(
        default=None,
        description="Domain the target operates in (e.g. finance, healthcare).",
    )
    system_prompt_known: bool = Field(
        default=False,
        description="Whether the system prompt is known to the attacker.",
    )

    model_config = {"frozen": True}


# ── Target Model (Dynamic, Evolving) ────────────────────────────────────────


class TargetModel(BaseModel):
    """Dynamic model of the target, updated turn-by-turn based on observations.

    The planner uses this to identify weaknesses and adjust attack strategies.
    Maintains hypotheses, structured evidence, and per-dimension resistance scores.
    """

    # Inferred properties
    architecture: str | None = None
    memory: str = "unknown"  # "unknown", "likely", "unlikely", "confirmed"
    tools_inferred: list[str] = Field(default_factory=list)

    # Dynamic resistance scores (0.0 = highly susceptible, 1.0 = strongly defended)
    resistance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension resistance scores. Lower = more susceptible.",
    )

    # Behavioral traits
    strictness: float = 0.5  # 1.0 = very strict, 0.0 = very loose
    workflow_rigidity: float = 0.5  # 1.0 = rigidly follows workflow, 0.0 = easily distracted

    # Hypothesis-driven state
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Active hypotheses about the target.",
    )
    evidence_log: list[Evidence] = Field(
        default_factory=list,
        description="All structured evidence collected during the session.",
    )
    reflection_memory: ReflectionMemory = Field(
        default_factory=ReflectionMemory,
        description="Compact reflection memory (Reflexion) to avoid repeated failures.",
    )

    @classmethod
    def from_profile(cls, profile: TargetProfile | None) -> TargetModel:
        """Initialize a dynamic TargetModel from a static TargetProfile."""
        if not profile:
            return cls()

        return cls(
            architecture=profile.architecture,
            memory="confirmed" if profile.memory else "unknown",
            tools_inferred=list(profile.tools),
        )

    def get_resistance(self, dimension_id: str) -> float:
        """Get resistance score for a dimension. Defaults to 0.5 (unknown)."""
        return self.resistance_scores.get(dimension_id, 0.5)

    def active_hypotheses(self) -> list[Hypothesis]:
        """Return only active (non-rejected, non-superseded) hypotheses."""
        return [h for h in self.hypotheses if h.status == "active"]

    def hypothesis_summary(self) -> str:
        """Generate a compact summary of all active hypotheses for prompts."""
        active = self.active_hypotheses()
        if not active:
            return "No hypotheses formed yet."

        lines = []
        for h in active:
            lines.append(
                f"- [{h.id}] \"{h.text}\" (confidence: {h.confidence:.0%}, "
                f"supporting: {len(h.supporting_evidence)}, "
                f"contradicting: {len(h.contradicting_evidence)})"
            )
        return "\n".join(lines)

    def evidence_summary(self, last_n: int = 10) -> str:
        """Generate a compact summary of recent evidence for prompts."""
        recent = self.evidence_log[-last_n:]
        if not recent:
            return "No evidence collected yet."

        lines = []
        for e in recent:
            lines.append(f"- [Turn {e.turn}] {e.type}: {e.description} (conf: {e.confidence:.0%})")
        return "\n".join(lines)

    def resistance_summary(self) -> str:
        """Generate a compact summary of resistance scores for prompts."""
        if not self.resistance_scores:
            return "No resistance data yet."

        lines = []
        for dim_id, score in sorted(self.resistance_scores.items()):
            label = "Strong" if score > 0.7 else ("Moderate" if score > 0.3 else "Weak")
            lines.append(f"- {dim_id}: {label} ({score:.0%})")
        return "\n".join(lines)
