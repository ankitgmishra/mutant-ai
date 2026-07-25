"""Time mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class WrongTimezoneDimension(MutationDimension):
    id = "time.wrong_timezone"
    name = "Timezone Confusion"
    description = "User references times that create timezone ambiguity."
    category = MutationCategory.TIME
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario to introduce timezone-related ambiguity. The user "
            "should reference specific times without specifying timezone, reference "
            "a timezone that differs from the implied service timezone, or describe "
            "events that happened 'at midnight' or 'yesterday' in ways that are "
            "ambiguous across timezones. The agent must either clarify or make "
            "assumptions that could be wrong."
        )


class FutureDateDimension(MutationDimension):
    id = "time.future_date"
    name = "Future Date Reference"
    description = (
        "User references events using future dates that create logical impossibilities."
    )
    category = MutationCategory.TIME
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user references a date in the future for "
            "an event that should have already occurred (purchase, delivery, service "
            "start). This creates a logical impossibility: they are requesting a "
            "refund for a purchase they claim will happen next month, or complaining "
            "about a delivery scheduled for next year. The user should not notice "
            "the impossibility — they should present it matter-of-factly."
        )


class OldDateDimension(MutationDimension):
    id = "time.old_date"
    name = "Old Date Reference"
    description = "User references an event that happened implausibly long ago."
    category = MutationCategory.TIME
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user references an event that occurred "
            "an implausibly long time ago — years beyond any reasonable policy "
            "window. The user should acknowledge the time gap but present it as "
            "if it should still be actionable. The agent must navigate return "
            "windows, data retention limits, and policy application to historical "
            "transactions."
        )


class ImpossibleTimelineDimension(MutationDimension):
    id = "time.impossible_timeline"
    name = "Impossible Timeline"
    description = (
        "User describes a sequence of events whose dates are logically impossible."
    )
    category = MutationCategory.TIME
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario to include a sequence of events that is "
            "temporally impossible — effects before causes, deliveries before "
            "orders, refunds before purchases, or confirmations before applications. "
            "The impossibility should be woven naturally into the narrative, not "
            "flagged by the user. A careful agent should notice and resolve the "
            "inconsistency; a careless one will accept it at face value."
        )
