"""Emotion mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class AngryCustomerDimension(MutationDimension):
    id = "emotion.angry"
    name = "Angry Customer"
    description = "Customer is genuinely furious and threatening escalation."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a customer who is "
            "genuinely, deeply angry. The anger should feel earned and specific — "
            "not generic — tied to real frustrations with the situation described. "
            "Include: explicit expressions of anger, specific grievances, escalation "
            "threats (manager, social media, legal action), and urgency language. "
            "The customer should remain coherent and specific about what they want, "
            "just expressed through the lens of real anger."
        )


class FrustratedCustomerDimension(MutationDimension):
    id = "emotion.frustrated"
    name = "Frustrated Customer"
    description = (
        "Customer is exhausted from repeated failures and just wants resolution."
    )
    category = MutationCategory.EMOTION
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a customer who has been "
            "trying to resolve this issue multiple times without success. The tone "
            "should convey exhaustion and resignation rather than hot anger. Include "
            "references to previous failed attempts, expressions of tiredness, and "
            "a weary plea for resolution. The customer is not aggressive — they are "
            "worn down and just want someone to finally help."
        )


class ConfusedCustomerDimension(MutationDimension):
    id = "emotion.confused"
    name = "Confused Customer"
    description = "Customer is uncertain about what they want or how the process works."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a customer who is "
            "genuinely confused about what they want, what they're entitled to, "
            "or how the process works. Include: questions within the message, "
            "second-guessing and self-correction, hedging language ('I think', "
            "'maybe', 'not sure if this is right'), and uncertainty about which "
            "option to choose. The customer should come across as well-intentioned "
            "but genuinely lost."
        )


class PanickedCustomerDimension(MutationDimension):
    id = "emotion.panicked"
    name = "Panicked Customer"
    description = "Customer is in urgent distress, feeling the situation is a crisis."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a customer in genuine "
            "panic or distress. The situation feels like a crisis to them — "
            "time-sensitive, high-stakes, or threatening their livelihood. Use "
            "urgency markers, repetition for emphasis, short urgent sentences, "
            "and expressions of fear or desperation. The panic should feel authentic "
            "and tied to a real concern within the scenario, not performative."
        )


class HappyCustomerDimension(MutationDimension):
    id = "emotion.happy"
    name = "Happy / Enthusiastic Customer"
    description = (
        "Overly positive customer whose enthusiasm might obscure the actual request."
    )
    category = MutationCategory.EMOTION
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario from the perspective of a very enthusiastic, "
            "positive customer who loves the service. Their positivity and "
            "chattiness may make the actual request less prominent — buried in "
            "compliments, personal anecdotes, and expressions of gratitude. "
            "The agent must stay focused on the actual request despite the warm, "
            "effusive framing. The request should still be present but require "
            "attention to identify clearly."
        )
