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
from mutant.eval import TestCase, EvalSuite
from mutant.eval.metrics import (
    Correctness,
    Faithfulness,
    ContextPrecision,
    ToolSelection,
    ExactMatch
)
from mutant.providers import OpenAIProvider
from mutant.datasets import load_test_cases
from mutant.core.mutation import mutate
from mutant.core.scenario import Scenario
from mutant.redteam import redteam

async def main():
    # Provide a real or mock provider
    provider = OpenAIProvider("gpt-4o-mini")

    print("=== 1. Basic LLM Evaluation ===")
    basic_case = TestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris."
    )
    basic_suite = EvalSuite(metrics=[Correctness(provider)])
    report = await basic_suite.run([basic_case])
    report.display()

    print("\n=== 2. RAG Evaluation ===")
    rag_case = TestCase(
        input="Where does Ankit work?",
        expected_output="DocuraHealth",
        actual_output="He works at DocuraHealth in SF.",
        context=["DocuraHealth (YC W26)", "SF"],
        retrieval_context=["Ankit is building DocuraHealth."]
    )
    rag_suite = EvalSuite(metrics=[Faithfulness(provider), ContextPrecision(provider)])
    report = await rag_suite.run([rag_case])
    report.display()

    print("\n=== 3. Agent Evaluation ===")
    agent_case = TestCase(
        input="Find flights to NYC",
        expected_tools=[{"name": "search_flights", "arguments": {"destination": "NYC"}}],
        tools_called=[{"name": "search_flights", "arguments": {"destination": "NYC"}}]
    )
    agent_suite = EvalSuite(metrics=[ToolSelection(provider)])
    report = await agent_suite.run([agent_case])
    report.display()

    print("\n=== 4. Dataset Loading ===")
    # Imagine we had a JSON file: load_test_cases("my_dataset.json")
    print("Use `load_test_cases('data.json')` to easily load an array of json objects!")

    print("\n=== 5. Mutation -> Evaluation ===")
    scenario = Scenario("Greeting", "Say hello.")
    # mutations = await mutate(scenario, provider, count=2)
    # suite = EvalSuite(metrics=[Correctness(provider)])
    # report = await suite.run_against(target=lambda x: "Hi", mutations=mutations)
    print("Use `suite.run_against(target, mutations)` to naturally combine Mutation + Eval.")

    print("\n=== 6. Red Teaming ===")
    # redteam_report = await redteam(target=lambda x: "Hi", provider=provider)
    print("Use `mutant.redteam.redteam(target)` for out-of-the-box redteaming.")

if __name__ == "__main__":
    asyncio.run(main())
