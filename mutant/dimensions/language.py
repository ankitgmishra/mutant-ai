"""Language mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class TypoDimension(MutationDimension):
    id = "language.typo"
    name = "Typo & Spelling Errors"
    description = (
        "User types with realistic spelling mistakes and autocorrect failures."
    )
    category = MutationCategory.LANGUAGE
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario with realistic typing errors that a real user "
            "might make on a phone or keyboard: transposed letters, missed letters, "
            "autocorrect substitutions, repeated letters, and phonetic misspellings. "
            "The errors should be subtle and plausible — not every word, roughly "
            "15–25% of words should have errors. The meaning must still be "
            "understandable. Do not correct them — preserve all errors."
        )


class MixedLanguageDimension(MutationDimension):
    id = "language.mixed_language"
    name = "Mixed Language (Code-Switching)"
    description = "Bilingual user switches between languages within a single message."
    category = MutationCategory.LANGUAGE
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user naturally switches between English and "
            "another language (choose one: Spanish, French, Hindi, Portuguese, or "
            "Arabic) within the same message — a realistic code-switching pattern "
            "seen among bilingual speakers. The switching should feel natural, not "
            "random. Key nouns, emotional expressions, and polite phrases are typical "
            "code-switch points. The overall message must remain understandable."
        )


class EmojiHeavyDimension(MutationDimension):
    id = "language.emoji"
    name = "Emoji Heavy Communication"
    description = "User communicates with heavy emoji usage alongside text."
    category = MutationCategory.LANGUAGE
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user uses emojis liberally throughout "
            "their message — after sentences, to express emotions, to replace "
            "words, and to emphasize points. The emoji usage should feel authentic "
            "to a mobile-native user, not robotic. The core message must be "
            "preserved and the request must still be clear despite the emoji density."
        )


class GrammarMistakeDimension(MutationDimension):
    id = "language.grammar_mistake"
    name = "Grammar Mistakes"
    description = "Non-native English speaker with grammatical errors."
    category = MutationCategory.LANGUAGE
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario as if written by a non-native English speaker "
            "who has intermediate proficiency. Introduce realistic grammatical "
            "patterns: missing articles (a/an/the), wrong verb tense, incorrect "
            "prepositions, subject-verb disagreement, and non-idiomatic phrasing. "
            "The errors should feel authentic to a real L2 English writer — not "
            "caricatured. The request must remain fully intelligible."
        )


class InformalSpeechDimension(MutationDimension):
    id = "language.informal_speech"
    name = "Informal / Slang Speech"
    description = "User writes in very casual, informal, or slang-heavy language."
    category = MutationCategory.LANGUAGE
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario in very informal, conversational language as if "
            "the user is texting a friend. Use contractions, abbreviations (tbh, "
            "ngl, asap, imo), casual vocabulary, sentence fragments, and informal "
            "openers. The tone should feel genuinely casual — not formal customer "
            "service language. The core request must remain clear."
        )
