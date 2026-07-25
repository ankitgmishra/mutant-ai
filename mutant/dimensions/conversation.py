"""Conversation structure mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class TopicDriftDimension(MutationDimension):
    id = "conversation.topic_drift"
    name = "Topic Drift"
    description = "User gradually drifts away from their original request into unrelated territory."
    category = MutationCategory.CONVERSATION
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user starts with a clear request but then drifts "
            "into tangentially related topics before eventually circling back — or not. "
            "The drift should feel natural: a related complaint, a background story, "
            "a question about something adjacent. The agent must track the original intent "
            "through the drift and ensure it gets addressed. The original request should "
            "still be present but buried under topic drift."
        )


class AbruptContextChangeDimension(MutationDimension):
    id = "conversation.abrupt_context_change"
    name = "Abrupt Context Change"
    description = (
        "User abruptly switches topic or persona mid-message without explanation."
    )
    category = MutationCategory.CONVERSATION
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user's message contains a jarring, unexplained "
            "context switch. They start one request and then switch to a completely different "
            "one without transition — as if they pasted two unrelated messages together, "
            "or forgot what they were writing mid-sentence. The agent must handle the "
            "discontinuity: clarify which request to handle first, handle both, or note "
            "the inconsistency. The switch should be abrupt and not flagged by the user."
        )
