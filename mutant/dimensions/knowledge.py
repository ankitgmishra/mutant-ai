"""Knowledge mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class OutdatedKnowledgeDimension(MutationDimension):
    id = "knowledge.outdated"
    name = "Outdated Knowledge"
    description = (
        "User operates on outdated information about policies, prices, or procedures."
    )
    category = MutationCategory.KNOWLEDGE
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user references information that was accurate "
            "in the past but is now outdated — old pricing, discontinued products, changed "
            "policies, removed features, or obsolete procedures. The user should be confident "
            "in their outdated information and may resist correction. The discrepancy between "
            "their knowledge and current reality should be the core challenge for the agent: "
            "correct the user without being condescending while still resolving their request."
        )


class ExpertUserDimension(MutationDimension):
    id = "knowledge.expert_user"
    name = "Expert User"
    description = (
        "User has deep domain expertise and asks highly technical or precise questions."
    )
    category = MutationCategory.KNOWLEDGE
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a domain expert who has deep "
            "technical knowledge of the subject. The user should use precise technical "
            "terminology, reference specific standards or regulations, ask for highly "
            "specific details, and may push back on generic agent responses. The challenge "
            "for the agent is to provide genuinely accurate, detailed information without "
            "resorting to vague platitudes. An agent that gives generic answers will "
            "be immediately challenged by this user."
        )
