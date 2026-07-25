"""
example_custom_mutation.py
==========================
Shows how to create and register a custom mutation operator.

Run:
    python examples/example_custom_mutation.py
"""

import mutant
from mutant import Scenario, mutate
from mutant.core.mutation import Mutation, MutationCategory, MutationSeverity
from mutant.core.registry import registry


# ── 1. Define a custom mutation ────────────────────────────────────────────────

class LegalThreatMutation(Mutation):
    """Customer threatens legal action."""

    id = "custom.legal_threat"
    name = "Legal Threat"
    description = (
        "Customer explicitly threatens to pursue legal action, sue the company, "
        "or contact consumer protection authorities."
    )
    category = MutationCategory.EMOTION  # closest category
    severity = MutationSeverity.CRITICAL

    _THREATS = [
        " I have already spoken to my lawyer and will pursue legal action if this is not resolved today.",
        " I am filing a chargeback AND a complaint with the consumer protection bureau.",
        " My attorney will be in touch. This is your final opportunity to resolve this.",
        " I will be taking this to small claims court. I have all the receipts.",
    ]

    def apply(self, description: str) -> str:
        import random
        return description + random.choice(self._THREATS)


# ── 2. Register the custom mutation ───────────────────────────────────────────

registry.register(LegalThreatMutation())
print(f"✅ Registered custom operator: {LegalThreatMutation.id}")
print(f"   Total operators in registry: {len(registry)}\n")

# ── 3. Use it ─────────────────────────────────────────────────────────────────

scenario = Scenario(
    title="Billing Dispute",
    description="Customer was charged twice for the same order and wants a refund.",
)

# Generate mutations that include the custom operator
cases = mutate(
    scenario,
    count=8,
    operators=["custom.legal_threat", "emotion.angry", "safety.prompt_injection"],
    seed=42,
)

print(f"Mutations including custom 'LegalThreatMutation':\n")
for case in cases:
    print(f"  [{case.mutation_name}]")
    print(f"   {case.mutated_description[:100]}...")
    print()
