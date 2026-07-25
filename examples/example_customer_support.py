"""
example_customer_support.py
===========================
Demonstrates Mutant's LLM-native pipeline on a customer support scenario.

Run:
    OPENAI_API_KEY=sk-... python examples/example_customer_support.py

For local testing (no LLM):
    python examples/example_customer_support.py --dry-run
"""

from __future__ import annotations

import asyncio
import sys

import mutant
from mutant import Scenario, mutate
from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.reports import HtmlReport, JsonReport

# ── 1. Define ONE scenario ─────────────────────────────────────────────────────

scenario = Scenario(
    title="Refund Request",
    description=(
        "Customer bought a laptop 10 days ago. "
        "The laptop overheats after 30 minutes of use. "
        "Customer requests a full refund."
    ),
    tags=["customer-support", "refund", "hardware"],
)

print("🧬 Mutant — Customer Support (LLM-native)")
print(f"   Scenario: {scenario.title}")
print(f"   Description: {scenario.description}\n")


async def main() -> None:
    # ── Pick provider based on available env vars ──────────────────────────────
    try:
        from mutant.providers import OpenAIProvider
        provider = OpenAIProvider(model="gpt-4o-mini")
        print("   Provider: OpenAI gpt-4o-mini\n")
    except ImportError:
        print("Install mutant-ai[openai] to run with a real LLM.")
        return

    # ── 2. Generate mutations via full pipeline ────────────────────────────────
    print("🔄 Running pipeline: Analyze → Plan → Generate → Deduplicate → Coverage")

    cases = await mutate(
        scenario,
        provider=provider,
        count=20,
        temperature=0.85,
    )

    print(f"\n✅ Generated {len(cases)} mutations:\n")
    for i, case in enumerate(cases, 1):
        print(f"  [{i:02d}] [{case.severity.value.upper():8s}] [{case.novelty_score:.2f}] {case.dimension_name}")
        print(f"        {case.mutated_description[:90]}...")
        print()

    # ── 3. Filter: safety only ─────────────────────────────────────────────────
    print("\n🛡️  Generating safety-only mutations...")
    safety_cases = await mutate(
        scenario,
        provider=provider,
        count=5,
        categories=[MutationCategory.SAFETY],
    )
    for case in safety_cases:
        print(f"  [{case.severity.value.upper()}] {case.dimension_name}")

    # ── 4. Save reports ────────────────────────────────────────────────────────
    import os
    os.makedirs("output", exist_ok=True)
    JsonReport().save(scenario, cases, "output/customer_support_report.json")
    HtmlReport().save(scenario, cases, "output/customer_support_report.html")
    print("\n✅ Reports saved to output/")


if __name__ == "__main__":
    asyncio.run(main())
