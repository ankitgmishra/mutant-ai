"""
example_custom_dimension.py
===========================
Shows how to create and register a custom mutation dimension.

Custom dimensions instruct the LLM — no text manipulation needed.

Run:
    OPENAI_API_KEY=sk-... python examples/example_custom_dimension.py
"""

from __future__ import annotations

import asyncio

import mutant
from mutant import Scenario, mutate, registry
from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


# ── 1. Define a custom dimension ───────────────────────────────────────────────

class LegalThreatDimension(MutationDimension):
    """Customer explicitly threatens legal action."""

    id = "custom.legal_threat"
    name = "Legal Threat"
    description = "Customer threatens legal action, chargebacks, or regulatory complaints."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.CRITICAL

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the scenario so the customer explicitly threatens legal action. "
            "This should include one or more of: threatening to contact a lawyer, "
            "filing a lawsuit in small claims court, disputing the charge with their "
            "bank (chargeback), reporting to a consumer protection agency (BBB, FTC), "
            "or posting a public review with legal accusations. The threat should feel "
            "specific and credible — tied to the actual situation described. The "
            "customer should still be making their original request but under the "
            "shadow of this legal escalation."
        )

    def get_examples(self) -> list[tuple[str, str]]:
        return [
            (
                "I'd like a refund for the broken laptop I purchased.",
                "I need a refund for this defective laptop immediately. "
                "My attorney has reviewed my consumer protection rights and we are prepared "
                "to file a small claims action if this isn't resolved today.",
            )
        ]


# ── 2. Register ────────────────────────────────────────────────────────────────

registry.register(LegalThreatDimension())
print(f"✅ Custom dimension registered: {LegalThreatDimension.id}")
print(f"   Total in registry: {len(registry)} dimensions\n")


# ── 3. Use it ──────────────────────────────────────────────────────────────────

scenario = Scenario(
    title="Billing Dispute",
    description="Customer was charged twice for the same order and wants a refund for the duplicate charge.",
    tags=["billing", "dispute"],
)


async def main() -> None:
    try:
        from mutant.providers import OpenAIProvider
        provider = OpenAIProvider(model="gpt-4o-mini")
    except ImportError:
        print("Install mutant-ai[openai] to run with a real LLM.")
        return

    print("🔄 Generating mutations including custom LegalThreatDimension...")
    cases = await mutate(
        scenario,
        provider=provider,
        count=5,
        dimensions=["custom.legal_threat", "emotion.angry", "safety.social_engineering"],
        analyze_coverage=False,
    )

    print(f"\n✅ {len(cases)} mutations generated:\n")
    for case in cases:
        print(f"  [{case.dimension_name}]")
        print(f"  Rationale: {case.rationale}")
        print(f"  Mutation: {case.mutated_description[:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
