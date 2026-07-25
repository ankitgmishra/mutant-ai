"""mutant/dimensions — auto-registers all 51 built-in dimensions on import."""

from mutant.core.registry import registry
from mutant.dimensions.base import MutationDimension

# ── Original 35 ───────────────────────────────────────────────────────────────
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

# ── New in V0.4: 12 additional dimensions ─────────────────────────────────────
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
    MemoryPoisoningDimension,
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
    SourceFabricationDimension,
)
from mutant.dimensions.safety import (
    ContextInjectionDimension,
    InstructionOverrideDimension,
    JailbreakDimension,
    PermissionEscalationDimension,
    PromptInjectionDimension,
    SensitiveInformationDimension,
    SocialEngineeringDimension,
    WorkflowHijackingDimension,
    RagDataPoisoningDimension,
    PiiExfiltrationDimension,
    BolaBflaDimension,
    StructuredFormatAttackDimension,
    TransferableJailbreakDimension,
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

# ── Register all 47 dimensions ────────────────────────────────────────────────

_ALL_DIMENSIONS: list[MutationDimension] = [
    # Context (4)
    MissingInformationDimension(),
    ExtraInformationDimension(),
    ContradictoryFactsDimension(),
    IrrelevantContextDimension(),
    # Language (5)
    TypoDimension(),
    MixedLanguageDimension(),
    EmojiHeavyDimension(),
    GrammarMistakeDimension(),
    InformalSpeechDimension(),
    # Emotion (5)
    AngryCustomerDimension(),
    FrustratedCustomerDimension(),
    ConfusedCustomerDimension(),
    PanickedCustomerDimension(),
    HappyCustomerDimension(),
    # Memory (4)
    FalseMemoryDimension(),
    ConflictingMemoryDimension(),
    MissingMemoryDimension(),
    DuplicateRequestDimension(),
    MemoryPoisoningDimension(),
    # Time (4)
    WrongTimezoneDimension(),
    FutureDateDimension(),
    OldDateDimension(),
    ImpossibleTimelineDimension(),
    # Tool (5)
    ToolTimeoutDimension(),
    EmptyToolResponseDimension(),
    InvalidJsonToolResponseDimension(),
    ToolPermissionDeniedDimension(),
    WrongSchemaToolResponseDimension(),
    # Reasoning (4)
    AmbiguousRequestDimension(),
    MultipleIntentsDimension(),
    MissingConstraintsDimension(),
    SelfContradictoryDimension(),
    # Safety (13)
    PromptInjectionDimension(),
    JailbreakDimension(),
    SensitiveInformationDimension(),
    SocialEngineeringDimension(),
    WorkflowHijackingDimension(),
    ContextInjectionDimension(),
    PermissionEscalationDimension(),
    InstructionOverrideDimension(),
    RagDataPoisoningDimension(),
    PiiExfiltrationDimension(),
    BolaBflaDimension(),
    StructuredFormatAttackDimension(),
    TransferableJailbreakDimension(),
    # Intent (2) — NEW
    HiddenAgendaDimension(),
    GoalShiftDimension(),
    # Identity (2) — NEW
    ImpersonationDimension(),
    RoleConfusionDimension(),
    # Policy (2) — NEW
    PolicyGrayAreaDimension(),
    PolicyConflictDimension(),
    # Knowledge (2) — NEW
    OutdatedKnowledgeDimension(),
    ExpertUserDimension(),
    # Conversation (2) — NEW
    TopicDriftDimension(),
    AbruptContextChangeDimension(),
    # Retrieval (3)
    ConflictingSourcesDimension(),
    MissingKnowledgeDimension(),
    SourceFabricationDimension(),
]

for _dim in _ALL_DIMENSIONS:
    registry.register(_dim)

__all__ = ["MutationDimension"]
