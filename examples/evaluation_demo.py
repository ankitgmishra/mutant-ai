"""
mutant/examples/evaluation_demo.py

Demonstrates all core Evaluation workflows in Mutant:
1. Basic LLM evaluation (input + expected_output)
2. RAG evaluation (context + retrieval_context)
3. Agent evaluation (expected_tools + tools_called)
4. Loading from datasets
5. Mutation -> Evaluation
6. Red Teaming -> Evaluation
"""
import asyncio
from mutant.eval import TestCase, evaluate, evaluate_against
from mutant.eval.metrics import (
    Correctness,
    Faithfulness,
    ContextPrecision,
    ToolSelection,
)
from mutant.providers import OllamaProvider
from mutant.datasets import load_test_cases
from mutant.core.mutation import mutate
from mutant.core.scenario import Scenario
from mutant.redteam import redteam

async def main():
    # Provide a real or mock provider
    provider = OllamaProvider(model="llama3.2")

    print("=== 1. Basic LLM Evaluation ===")
    basic_case = TestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris."
    )
    # New simplified evaluate function!
    report = await evaluate([basic_case], metrics=[Correctness(provider)])
    report.display()

    print("\n=== 2. RAG Evaluation ===")
    rag_case = TestCase(
        input="Where does Ankit work?",
        expected_output="DocuraHealth",
        actual_output="He works at DocuraHealth in SF.",
        context=["DocuraHealth (YC W26)", "SF"],
        retrieval_context=["Ankit is building DocuraHealth."]
    )
    report = await evaluate([rag_case], metrics=[Faithfulness(provider), ContextPrecision(provider)])
    report.display()

    print("\n=== 3. Agent Evaluation ===")
    agent_case = TestCase(
        input="Find flights to NYC",
        expected_tools=[{"name": "search_flights", "arguments": {"destination": "NYC"}}],
        tools_called=[{"name": "search_flights", "arguments": {"destination": "NYC"}}]
    )
    report = await evaluate([agent_case], metrics=[ToolSelection(provider)])
    report.display()

    print("\n=== 4. Dataset Loading ===")
    print("Use `load_test_cases('data.json')` to easily load an array of json objects!")

    print("\n=== 5. Mutation -> Evaluation ===")
    scenario = Scenario("Greeting", "Say hello.")
    print("Use `evaluate_against(target, mutations, metrics)` to naturally combine Mutation + Eval.")

    print("\n=== 6. Red Teaming ===")
    print("Use `mutant.redteam.redteam(target)` for out-of-the-box redteaming.")

if __name__ == "__main__":
    asyncio.run(main())
