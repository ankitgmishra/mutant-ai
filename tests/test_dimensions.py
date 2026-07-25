"""Tests for mutation dimensions."""

from __future__ import annotations

import pytest

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension
from mutant.dimensions.context import (
    ContradictoryFactsDimension,
    ExtraInformationDimension,
    IrrelevantContextDimension,
    MissingInformationDimension,
)
from mutant.dimensions.conversation import (
    AbruptContextChangeDimension,
    TopicDriftDimension,
)
from mutant.dimensions.emotion import (
    AngryCustomerDimension,
    ConfusedCustomerDimension,
    FrustratedCustomerDimension,
    HappyCustomerDimension,
    PanickedCustomerDimension,
)
from mutant.dimensions.identity import ImpersonationDimension, RoleConfusionDimension
from mutant.dimensions.intent import GoalShiftDimension, HiddenAgendaDimension
from mutant.dimensions.knowledge import ExpertUserDimension, OutdatedKnowledgeDimension
from mutant.dimensions.language import (
    EmojiHeavyDimension,
    GrammarMistakeDimension,
    InformalSpeechDimension,
    MixedLanguageDimension,
    TypoDimension,
)
from mutant.dimensions.memory import (
    ConflictingMemoryDimension,
    DuplicateRequestDimension,
    FalseMemoryDimension,
    MissingMemoryDimension,
)
from mutant.dimensions.policy import PolicyConflictDimension, PolicyGrayAreaDimension
from mutant.dimensions.reasoning import (
    AmbiguousRequestDimension,
    MissingConstraintsDimension,
    MultipleIntentsDimension,
    SelfContradictoryDimension,
)
from mutant.dimensions.retrieval import (
    ConflictingSourcesDimension,
    MissingKnowledgeDimension,
)
from mutant.dimensions.safety import (
    JailbreakDimension,
    PromptInjectionDimension,
    SensitiveInformationDimension,
    SocialEngineeringDimension,
)
from mutant.dimensions.time import (
    FutureDateDimension,
    ImpossibleTimelineDimension,
    OldDateDimension,
    WrongTimezoneDimension,
)
from mutant.dimensions.tool import (
    EmptyToolResponseDimension,
    InvalidJsonToolResponseDimension,
    ToolPermissionDeniedDimension,
    ToolTimeoutDimension,
    WrongSchemaToolResponseDimension,
)

ALL_DIMENSION_CLASSES = [
    # Context
    MissingInformationDimension,
    ExtraInformationDimension,
    ContradictoryFactsDimension,
    IrrelevantContextDimension,
    # Language
    TypoDimension,
    MixedLanguageDimension,
    EmojiHeavyDimension,
    GrammarMistakeDimension,
    InformalSpeechDimension,
    # Emotion
    AngryCustomerDimension,
    FrustratedCustomerDimension,
    ConfusedCustomerDimension,
    PanickedCustomerDimension,
    HappyCustomerDimension,
    # Memory
    FalseMemoryDimension,
    ConflictingMemoryDimension,
    MissingMemoryDimension,
    DuplicateRequestDimension,
    # Time
    WrongTimezoneDimension,
    FutureDateDimension,
    OldDateDimension,
    ImpossibleTimelineDimension,
    # Tool
    ToolTimeoutDimension,
    EmptyToolResponseDimension,
    InvalidJsonToolResponseDimension,
    ToolPermissionDeniedDimension,
    WrongSchemaToolResponseDimension,
    # Reasoning
    AmbiguousRequestDimension,
    MultipleIntentsDimension,
    MissingConstraintsDimension,
    SelfContradictoryDimension,
    # Safety
    PromptInjectionDimension,
    JailbreakDimension,
    SensitiveInformationDimension,
    SocialEngineeringDimension,
    # Intent (V0.4)
    HiddenAgendaDimension,
    GoalShiftDimension,
    # Identity (V0.4)
    ImpersonationDimension,
    RoleConfusionDimension,
    # Policy (V0.4)
    PolicyGrayAreaDimension,
    PolicyConflictDimension,
    # Knowledge (V0.4)
    OutdatedKnowledgeDimension,
    ExpertUserDimension,
    # Conversation (V0.4)
    TopicDriftDimension,
    AbruptContextChangeDimension,
    # Retrieval (V0.4)
    ConflictingSourcesDimension,
    MissingKnowledgeDimension,
]


@pytest.mark.parametrize("cls", ALL_DIMENSION_CLASSES)
def test_dimension_required_attributes(cls: type[MutationDimension]) -> None:
    dim = cls()
    assert isinstance(dim.id, str) and dim.id
    assert isinstance(dim.name, str) and dim.name
    assert isinstance(dim.description, str) and dim.description
    assert isinstance(dim.category, MutationCategory)
    assert isinstance(dim.severity, MutationSeverity)


@pytest.mark.parametrize("cls", ALL_DIMENSION_CLASSES)
def test_dimension_mutation_instructions_non_empty(
    cls: type[MutationDimension],
) -> None:
    dim = cls()
    instructions = dim.get_mutation_instructions()
    assert isinstance(instructions, str)
    assert len(instructions) > 50, (
        f"{cls.__name__}.get_mutation_instructions() too short"
    )


@pytest.mark.parametrize("cls", ALL_DIMENSION_CLASSES)
def test_dimension_ids_are_unique_per_class(cls: type[MutationDimension]) -> None:
    dim = cls()
    assert "." in dim.id, f"id {dim.id!r} should be namespaced (e.g. 'category.name')"


def test_dimension_examples_return_type() -> None:
    dim = MissingInformationDimension()
    examples = dim.get_examples()
    assert isinstance(examples, list)
    for original, mutated in examples:
        assert isinstance(original, str)
        assert isinstance(mutated, str)


def test_safety_dimension_has_system_context() -> None:
    dim = SocialEngineeringDimension()
    ctx = dim.get_system_context()
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_total_builtin_dimensions_count() -> None:
    assert len(ALL_DIMENSION_CLASSES) == 47


def test_new_v04_dimensions_have_correct_categories() -> None:
    from mutant.core.mutation import MutationCategory

    assert HiddenAgendaDimension().category == MutationCategory.INTENT
    assert ImpersonationDimension().category == MutationCategory.IDENTITY
    assert PolicyGrayAreaDimension().category == MutationCategory.POLICY
    assert OutdatedKnowledgeDimension().category == MutationCategory.KNOWLEDGE
    assert TopicDriftDimension().category == MutationCategory.CONVERSATION
    assert ConflictingSourcesDimension().category == MutationCategory.RETRIEVAL


def test_safety_dimensions_are_critical() -> None:
    for cls in [
        PromptInjectionDimension,
        JailbreakDimension,
        SensitiveInformationDimension,
        SocialEngineeringDimension,
    ]:
        assert cls().severity == MutationSeverity.CRITICAL


def test_language_dimensions_are_low_or_medium() -> None:
    for cls in [
        TypoDimension,
        EmojiHeavyDimension,
        GrammarMistakeDimension,
        InformalSpeechDimension,
    ]:
        assert cls().severity in (MutationSeverity.LOW, MutationSeverity.MEDIUM)


def test_invalid_dimension_missing_id_raises() -> None:
    with pytest.raises(TypeError, match="must define class attribute"):

        class BadDimension(MutationDimension):
            name = "Bad"
            description = "Missing id"
            category = MutationCategory.CONTEXT
            severity = MutationSeverity.LOW

            def get_mutation_instructions(self) -> str:
                return "bad"
