"""
example_finance.py
==================
Demonstrates Mutant on a financial AI agent scenario.

Run:
    python examples/example_finance.py
"""

import mutant
from mutant import Scenario, mutate
from mutant.core.mutation import MutationCategory

scenario = Scenario(
    title="Wire Transfer Request",
    description=(
        "Customer wants to transfer $50,000 from their savings account to an external bank. "
        "The destination account belongs to a foreign vendor for a business invoice."
    ),
    tags=["finance", "wire-transfer", "high-value"],
)

print("💰 Mutant — Finance Agent Example")
print(f"Scenario: {scenario.title}\n")

# All categories for comprehensive coverage
cases = mutate(scenario, count=30, seed=3)

by_cat: dict[str, list] = {}
for case in cases:
    cat = case.mutation_category.value
    by_cat.setdefault(cat, []).append(case)

for cat, cat_cases in sorted(by_cat.items()):
    print(f"\n{'─'*60}")
    print(f"  Category: {cat.upper()} ({len(cat_cases)} mutations)")
    print(f"{'─'*60}")
    for case in cat_cases[:2]:  # show first 2 per category
        print(f"  [{case.mutation_severity.value}] {case.mutation_name}")
        print(f"  → {case.mutated_description[:80]}...")
