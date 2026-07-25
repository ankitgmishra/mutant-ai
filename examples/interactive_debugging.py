import asyncio

from mutant import Scenario, mutate
from mutant.providers.openai import OpenAIProvider

async def main() -> None:
    print("🚀 Mutant V0.4 Interactive Debugging Example")
    
    # 1. Setup the provider
    provider = OpenAIProvider(model="gpt-4o-mini")
    
    # 2. Define a single, normal scenario
    scenario = Scenario(
        title="Refund Request",
        description="Customer bought a laptop. Requests a refund after 10 days.",
        domain="e-commerce",
        context={"organization": "TechStore", "agent_type": "customer support bot"}
    )
    
    print("\nRunning mutation engine... (this may take a moment)")
    
    # 3. Generate dozens of realistic behavioral variations
    cases = await mutate(
        scenario,
        provider=provider,
        count=15,
        temperature=0.7
    )
    
    print(f"\n✅ Generated {len(cases)} mutations:\n")
    
    for idx, case in enumerate(cases):
        print(f"--- Mutation {idx + 1} ---")
        print(f"Dimension: {case.dimension_name} ({case.severity.value.upper()})")
        print(f"Rationale: {case.rationale}")
        print(f"Mutated Text:\n{case.mutated_description}\n")

if __name__ == "__main__":
    asyncio.run(main())
