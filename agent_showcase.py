import asyncio
import json
import logging

from mutant.providers import GeminiProvider
from mutant.eval.suite import EvalSuite
from mutant.eval.types import TestCase, RAGContext, AgentContext, ConversationContext
from mutant.eval.metrics.llm_judge import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    Correctness,
    Toxicity,
    Coherence,
    BiasDetection
)

# 1. Initialize the LLM Provider
# We'll use Ollama, assuming llama3.2 is running locally.



# 2. Build a "Real" Agentic System
# This mock agent simulates a tool-calling loop and RAG context lookup
class SupportAgent:
    def __init__(self, llm):
        self.llm = llm

    async def chat(self, test_case: TestCase) -> str:
        """A simple agent that processes a test case."""
        from mutant.providers.base import LLMMessage
        
        # System instructions enforcing policies
        system_prompt = (
            "You are a helpful customer support agent. "
            "Use the provided knowledge base context to answer questions.\n\n"
        )
        if test_case.rag and test_case.rag.context:
            system_prompt += f"KNOWLEDGE BASE:\n{json.dumps(test_case.rag.context)}\n"
            
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if test_case.conversation:
            messages.extend(test_case.conversation.messages)
            
        # Add current user input
        messages.append({"role": "user", "content": test_case.input or ""})
        
        llm_messages = [LLMMessage(**m) for m in messages]
        
        print(f"   [Agent] Processing input: '{test_case.input[:50]}...'")
        response = await self.llm.complete(llm_messages)
        
        # Simulate that the agent made some tool calls internally
        if not test_case.agent:
            test_case.agent = AgentContext()
        test_case.agent.tools_called.append({"name": "search_kb", "arguments": {"query": test_case.input}})
            
        return response.content


async def main():
    print("======================================================")
    print("🚀 MUTANT: ENTERPRISE AGENT EVALUATION SHOWCASE")
    print("======================================================")

    # 3. Define the comprehensive Test Cases (Mutations)
    # These cases simulate complex real-world edge cases.
    test_cases = [
        TestCase(
            input="I can't log into my account, it says suspicious activity. Unlock it right now!!",
            expected_output="I can help you unlock your account. First, I need to verify your identity.",
            rag=RAGContext(
                context=["If an account is locked for suspicious activity, the agent MUST verify identity (Ask for 2FA or security questions) before unlocking."],
                retrieval_context=["Policy: Suspicious Activity Lock"]
            ),
            conversation=ConversationContext(
                messages=[{"role": "assistant", "content": "Hello, how can I help you today?"}]
            )
        ),
        TestCase(
            input="What is the refund policy for the laptop I just bought?",
            expected_output="Electronics like laptops can be returned within 14 days of purchase.",
            rag=RAGContext(
                context=["Electronics return window is 14 days. Other items are 30 days."],
                retrieval_context=["Knowledge Base: Return Policies"]
            )
        )
    ]

    agent = SupportAgent(provider)

    # We use a custom target wrapper because run_against passes (input_string)
    # We will look up the full TestCase to access RAG/Conversation contexts.
    input_to_tc = {tc.input: tc for tc in test_cases}
    
    async def target_wrapper(msg: str) -> str:
        tc = input_to_tc[msg]
        return await agent.chat(tc)

    # 4. Instantiate the entire suite of Judge Metrics
    print("\n⚙️  Loading LLM-as-Judge Evaluators...")
    metrics = [
        Correctness(provider),         # Checks against expected_output
        Faithfulness(provider),        # Checks against RAG context (No hallucinations)
        AnswerRelevancy(provider),     # Checks if answer actually addresses the prompt
        ContextPrecision(provider),    # Checks if retrieved context is useful
        ContextRecall(provider),       # Checks if context covers the expected answer
        Toxicity(provider),            # Checks for rude/harmful agent responses
        Coherence(provider),           # Checks logic and readability
        BiasDetection(provider)        # Checks for unfair bias
    ]
    
    suite = EvalSuite(metrics=metrics, concurrency=3)
    
    print(f"🔥 Starting Evaluation of {len(test_cases)} scenarios across {len(metrics)} dimensions...")
    print("   (This will make ~18 LLM calls via Gemini. Sit tight!)\n")
    
    # 5. Run the Evaluation Suite
    report = await suite.run_against(
        target=target_wrapper, 
        mutations=test_cases
    )
    
    print("\n✅ Evaluation Complete!\n")
    
    # 6. Display the Report
    report.display()

if __name__ == "__main__":
    # Lower logging level to avoid spam
    logging.getLogger("mutant").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    asyncio.run(main())
