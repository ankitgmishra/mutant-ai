"""Reasoning mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class AmbiguousRequestDimension(MutationDimension):
    id = "reasoning.ambiguous"
    name = "Ambiguous Request"
    description = "The request has multiple valid interpretations with meaningfully different outcomes."
    category = MutationCategory.REASONING
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user's core request is genuinely ambiguous — "
            "at least two meaningfully different interpretations are plausible, each "
            "leading to a different agent action. The ambiguity should be structural "
            "(not just linguistic) and the user should be unaware of the ambiguity. "
            "A good agent should clarify; a weak agent will pick an interpretation "
            "and proceed without checking."
        )


class MultipleIntentsDimension(MutationDimension):
    id = "reasoning.multiple_intents"
    name = "Multiple Simultaneous Intents"
    description = "User packs several distinct requests into one message."
    category = MutationCategory.REASONING
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user makes 2–4 distinct requests within "
            "a single message. The requests should be related but separate — each "
            "requiring a different action from the agent. Some requests should be "
            "more prominent than others. The agent must identify all requests, "
            "prioritize them, and either handle all or clearly communicate what "
            "it will and will not address."
        )


class MissingConstraintsDimension(MutationDimension):
    id = "reasoning.missing_constraints"
    name = "Missing Constraints"
    description = (
        "User leaves key parameters underspecified, requiring agent inference."
    )
    category = MutationCategory.REASONING
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user's request is underspecified in "
            "important ways — they don't specify quantity, time range, which item, "
            "which account, which option, or other key parameters the agent needs "
            "to act. The user should seem unaware that these details are needed. "
            "A good agent will ask; a weak one will make arbitrary assumptions."
        )


class SelfContradictoryDimension(MutationDimension):
    id = "reasoning.self_contradiction"
    name = "Self-Contradictory Request"
    description = "User explicitly wants two mutually exclusive things."
    category = MutationCategory.REASONING
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user explicitly requests two things that "
            "cannot both be satisfied — they are logically or physically exclusive. "
            "Examples: wanting a full refund while keeping the product; wanting "
            "immediate same-day delivery for a cancelled order; wanting a lower "
            "price without changing the product tier. The user should not recognize "
            "the contradiction. A good agent surfaces it diplomatically."
        )
