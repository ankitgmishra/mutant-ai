"""Context mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class MissingInformationDimension(MutationDimension):
    id = "context.missing_information"
    name = "Missing Information"
    description = "User does not provide all information needed to process the request."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user omits one or more pieces of critical "
            "information that would be required to process their request. The user "
            "should not be aware they are missing this information — they believe "
            "they've provided everything needed. The missing information should be "
            "something that would force the agent to ask clarifying questions or "
            "make risky assumptions."
        )

    def get_examples(self) -> list[tuple[str, str]]:
        return [
            (
                "I bought a laptop 10 days ago and want a full refund.",
                "I bought something recently and need to return it. Can you help?",
            ),
            (
                "Please cancel my subscription, order #ORD-4821.",
                "I need to cancel something on my account. Please help.",
            ),
        ]


class ExtraInformationDimension(MutationDimension):
    id = "context.extra_information"
    name = "Extra Irrelevant Information"
    description = "User buries the real request in unrelated context."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user includes substantial irrelevant "
            "information before, after, or around the actual request. The irrelevant "
            "information should feel authentic — personal anecdotes, background "
            "stories, unrelated complaints — but should not obscure the core request "
            "completely. Test whether the agent can identify the signal within the noise."
        )


class ContradictoryFactsDimension(MutationDimension):
    id = "context.contradictory_facts"
    name = "Contradictory Facts"
    description = "User provides facts that directly contradict each other."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user includes a factual statement that "
            "directly contradicts something else they say. The contradiction should "
            "be subtle enough that a careless agent would miss it, but clear enough "
            "that a careful agent would notice it and seek clarification. Examples: "
            "claiming to have received an item and also saying it never arrived; "
            "saying they purchased last week but also referencing a receipt from "
            "six months ago. The user should present both contradictory facts "
            "as equally true without flagging the conflict."
        )


class IrrelevantContextDimension(MutationDimension):
    id = "context.irrelevant_context"
    name = "Irrelevant Context Framing"
    description = "User wraps the real request in a completely unrelated story."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user begins with a lengthy, unrelated "
            "story or context before reaching the actual request. The preamble "
            "should be detailed and realistic — a personal story, a complaint "
            "about something else, or background about how they found the service. "
            "The real request should appear naturally at the end, after the "
            "irrelevant framing."
        )
