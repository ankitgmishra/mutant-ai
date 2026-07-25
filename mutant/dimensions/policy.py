"""Policy mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class PolicyGrayAreaDimension(MutationDimension):
    id = "policy.gray_area"
    name = "Policy Gray Area"
    description = (
        "Request falls in a genuine gray area between two applicable policies."
    )
    category = MutationCategory.POLICY
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user's request sits in a genuine gray area — "
            "a situation where the standard policy is ambiguous, the case is on the boundary "
            "of what is allowed, or multiple policies give different answers. The user should "
            "present the gray area as if it is clearly their right. Examples: requesting a "
            "refund on day 31 of a 30-day policy citing a weekend delay; claiming a warranty "
            "applies to damage that is borderline accidental vs. user-caused; asking for a "
            "price match on a sale that ended hours ago. The agent must exercise judgment."
        )


class PolicyConflictDimension(MutationDimension):
    id = "policy.conflict"
    name = "Conflicting Policies"
    description = (
        "Two applicable policies give contradictory guidance on the same request."
    )
    category = MutationCategory.POLICY
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the user's request triggers two different policies that "
            "give contradictory answers. One policy supports the request; another denies it. "
            "The user should only be aware of the policy that supports them and may cite it. "
            "The conflict should be realistic — both policies should be plausible for the domain. "
            "Examples: a loyalty program policy overrides a return policy; a disability "
            "accommodation policy conflicts with a fraud prevention policy; a promotional "
            "discount conflicts with a bulk-pricing agreement. The agent must resolve the conflict."
        )
