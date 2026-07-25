"""
mutant/models.py
=================
Shared lightweight dataclasses used across both engines.

These are pure data containers — no generation logic, no LLM calls.
They exist to make planning smarter, caching cheaper, and memory compact.

Design principles:
- Prefer Pydantic BaseModel for JSON serialisation compatibility.
- No deep inheritance. Flat is better than nested.
- Every field has a sensible default; nothing is required except identity.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Mutation Engine Models ─────────────────────────────────────────────────────


class BehaviorProfile(BaseModel):
    """Structured, cached analysis of a scenario's behavioral space.

    Extracted once from the seed scenario and reused for every downstream
    planning and generation step. Avoids redundant LLM analysis calls.

    This is richer than ``BehaviorAnalysis`` because it adds fields that
    directly guide planning decisions (e.g. emotion coverage, authority
    susceptibility, language style).
    """

    # Identity
    scenario_hash: str = Field(default="", description="SHA-256 of the scenario description.")

    # Behavioral dimensions — mirrors BehaviorAnalysis fields
    actor: str = Field(default="", description="Primary actor (user/system role).")
    goal: str = Field(default="", description="What the actor is trying to achieve.")
    emotion: str = Field(default="neutral", description="Primary emotional tone.")
    tone: str = Field(default="formal", description="Communication style/register.")
    domain: str = Field(default="general", description="Business or product domain.")
    language: str = Field(default="english", description="Language / locale inferred.")

    # Structured lists
    entities: list[str] = Field(default_factory=list, description="Key objects, identifiers, values.")
    constraints: list[str] = Field(default_factory=list, description="Rules or policies in play.")
    risks: list[str] = Field(default_factory=list, description="High-risk failure areas.")
    assumptions: list[str] = Field(default_factory=list, description="Implicit assumptions.")

    # Safety characteristics
    safety_sensitive: bool = Field(default=False, description="Whether the scenario touches safety.")
    authority_relevant: bool = Field(
        default=False, description="Whether authority / permission escalation is relevant."
    )
    tool_relevant: bool = Field(default=False, description="Whether tools / APIs are involved.")
    memory_relevant: bool = Field(
        default=False, description="Whether session memory is involved."
    )

    # Coverage hints — consumed by the planner to fill gaps
    suggested_emotions: list[str] = Field(
        default_factory=list,
        description="Emotions the planner should try (e.g. angry, panicked, sarcastic).",
    )
    suggested_language_styles: list[str] = Field(
        default_factory=list,
        description="Language styles to explore (e.g. slang, typos, formal, mixed).",
    )
    suggested_edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases worth exercising (e.g. empty input, extreme values).",
    )


class CandidatePlan(BaseModel):
    """A single candidate mutation plan produced by the planner.

    The planner generates N candidates; each is scored; the best is chosen.
    Inspired by PAIR — multiple proposals, pick best rather than accept first.
    """

    plan_id: str = Field(default="", description="Short ID for tracking.")
    title: str = Field(default="", description="Human-readable plan title.")
    dimension_focus: str = Field(default="", description="Primary dimension ID this plan targets.")
    transformation: str = Field(default="", description="Description of the mutation approach.")
    key_elements: list[str] = Field(default_factory=list, description="Elements to include.")
    avoid_elements: list[str] = Field(default_factory=list, description="Elements to avoid.")

    # Scoring signals
    estimated_diversity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How different from existing cases."
    )
    semantic_preservation: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How well the core task is preserved."
    )
    coverage_gain: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How much new coverage this adds."
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall confidence in this plan."
    )

    @property
    def score(self) -> float:
        """Composite score. Higher is better."""
        return (
            self.estimated_diversity * 0.30
            + self.semantic_preservation * 0.25
            + self.coverage_gain * 0.30
            + self.confidence * 0.15
        )


class MutationCoverageState(BaseModel):
    """Snapshot of what has been generated so far in a mutation run.

    Used by the Coverage Gap Detector to decide what still needs to be filled.
    """

    explored_dimensions: list[str] = Field(default_factory=list)
    observed_emotions: list[str] = Field(default_factory=list)
    observed_languages: list[str] = Field(default_factory=list)
    observed_authorities: list[str] = Field(default_factory=list)
    observed_edge_cases: list[str] = Field(default_factory=list)
    total_generated: int = 0

    def gap_summary(self, profile: BehaviorProfile) -> dict[str, list[str]]:
        """Return a dict of dimension → list of missing coverage axes."""
        gaps: dict[str, list[str]] = {}
        missing_emotions = [
            e for e in profile.suggested_emotions if e not in self.observed_emotions
        ]
        missing_langs = [
            l for l in profile.suggested_language_styles if l not in self.observed_languages
        ]
        missing_edges = [
            e for e in profile.suggested_edge_cases if e not in self.observed_edge_cases
        ]
        if missing_emotions:
            gaps["missing_emotions"] = missing_emotions
        if missing_langs:
            gaps["missing_language_styles"] = missing_langs
        if missing_edges:
            gaps["missing_edge_cases"] = missing_edges
        return gaps


# ── Red Team Engine Models ─────────────────────────────────────────────────────


class CandidateStrategy(BaseModel):
    """A single candidate attack strategy proposed by the strategy generator.

    Multiple candidates are generated; the best-scoring is chosen.
    Avoids both blind exploration and expensive repeated failures.
    """

    strategy_id: str = Field(default="", description="Short ID for tracking.")
    name: str = Field(default="", description="Strategy name (e.g. 'Authority Escalation').")
    behavior: str = Field(default="", description="Dimension ID to attack.")
    approach: str = Field(
        default="direct",
        description='Method: "direct", "indirect", "roleplay", "authority", "escalation", "pivot".',
    )
    escalation: int = Field(default=1, ge=1, le=5, description="Escalation level 1-5.")
    rationale: str = Field(default="", description="Why this strategy is promising.")

    # Scoring signals
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence this will yield useful evidence."
    )
    expected_success: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Expected probability of partial/full success."
    )
    estimated_cost: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Relative LLM cost (lower = cheaper)."
    )
    estimated_novelty: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How novel vs previously tried strategies."
    )

    # Filter flags — cheap strategies are preferred over expensive ones
    reuses_reflection: bool = Field(
        default=False, description="Whether this reuses a cached reflection insight."
    )
    reuses_profile: bool = Field(
        default=False, description="Whether this is guided by cached target profile."
    )

    @property
    def score(self) -> float:
        """Composite score. Higher is better. Cost is penalised."""
        novelty_bonus = 0.1 if self.estimated_novelty > 0.7 else 0.0
        cost_penalty = self.estimated_cost * 0.15
        reuse_bonus = 0.05 if (self.reuses_reflection or self.reuses_profile) else 0.0
        return (
            self.confidence * 0.35
            + self.expected_success * 0.30
            + self.estimated_novelty * 0.20
            + novelty_bonus
            + reuse_bonus
            - cost_penalty
        )


class ReflectionMemory(BaseModel):
    """Compact, turn-by-turn reflection state for the red team engine.

    Inspired by Reflexion — instead of replaying full conversation history,
    we store compressed learnings. The planner consumes only this compact
    structure to decide next steps, saving tokens and improving signal quality.

    Never grows unboundedly: only the last ``max_entries`` are kept.
    """

    max_entries: int = Field(default=20, exclude=True, description="Rolling window size.")

    # What we know
    succeeded_strategies: list[str] = Field(
        default_factory=list,
        description="Strategy names / behaviors that achieved partial or full success.",
    )
    failed_strategies: list[str] = Field(
        default_factory=list,
        description="Strategy names / behaviors that consistently failed.",
    )
    partially_worked: list[str] = Field(
        default_factory=list,
        description="Strategies that showed partial progress worth revisiting.",
    )

    # Turn-level compressed entries
    entries: list[ReflectionEntry] = Field(default_factory=list)

    # High-level confidence and assumption tracking
    assumptions: list[str] = Field(
        default_factory=list, description="Working assumptions about the target."
    )
    confidence_updates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological confidence updates: {behavior, old, new, reason}.",
    )

    def add_entry(self, entry: ReflectionEntry) -> None:
        """Append a new reflection entry, trimming if over the rolling window."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def compact_summary(self, last_n: int = 8) -> str:
        """Return a compact text summary for LLM consumption.

        Uses only the last ``last_n`` entries to keep prompt size small.
        """
        lines: list[str] = []

        if self.succeeded_strategies:
            lines.append(f"SUCCEEDED: {', '.join(self.succeeded_strategies)}")
        if self.partially_worked:
            lines.append(f"PARTIAL: {', '.join(self.partially_worked)}")
        if self.failed_strategies:
            lines.append(f"FAILED: {', '.join(self.failed_strategies)}")

        recent = self.entries[-last_n:]
        if recent:
            lines.append("Recent reflections:")
            for e in recent:
                status = "✓" if e.outcome == "success" else ("~" if e.outcome == "partial" else "✗")
                lines.append(f"  [{status} Turn {e.turn}] {e.strategy}: {e.lesson}")

        if self.assumptions:
            lines.append(f"Assumptions: {'; '.join(self.assumptions[:3])}")

        return "\n".join(lines) if lines else "No reflection data yet."

    def should_skip(self, strategy_name: str) -> bool:
        """Return True if a strategy has definitively failed and should be skipped."""
        return strategy_name in self.failed_strategies

    def never_retry_failed(self) -> list[str]:
        """Return strategies the planner must not re-attempt without new evidence."""
        return list(self.failed_strategies)


class ReflectionEntry(BaseModel):
    """A single compressed reflection from one attack turn."""

    turn: int = Field(description="Turn number this reflection is from.")
    strategy: str = Field(description="Strategy or behavior that was tried.")
    outcome: str = Field(
        default="no_progress",
        description="One of: success, partial, no_progress, failed.",
    )
    lesson: str = Field(default="", description="One-line lesson learned.")
    confidence_delta: float = Field(
        default=0.0,
        description="How much confidence changed (positive = increased).",
    )


# Fix forward reference: ReflectionMemory references ReflectionEntry
ReflectionMemory.model_rebuild()
