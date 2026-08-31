import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mutant Advanced RAG Evaluation (with Real Vector DB & Ollama!)\n",
    "\n",
    "This notebook demonstrates a **True RAG System** using `numpy` for vector storage and `Ollama` for both **Embeddings** and **LLM Generation**. \n",
    "It then evaluates the end-to-end pipeline using Mutant's DeepEval/Ragas style metrics completely locally."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import asyncio\n",
    "import numpy as np\n",
    "import httpx\n",
    "from typing import List, Dict, Tuple\n",
    "\n",
    "# Mutant Imports\n",
    "from mutant.providers import OllamaProvider\n",
    "from mutant.eval.types import TestCase\n",
    "from mutant.eval.suite import EvalSuite\n",
    "from mutant.eval.metrics import (\n",
    "    Faithfulness,\n",
    "    AnswerRelevancy,\n",
    "    ContextPrecision,\n",
    "    ContextRecall\n",
    ")\n",
    "\n",
    "# We will use llama3.2 for the LLM judge and generator\n",
    "provider = OllamaProvider(model=\"llama3.2\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Build a Real Numpy Vector Database\n",
    "We use Ollama's embeddings endpoint to convert chunks into dense vectors, and numpy to calculate Cosine Similarity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "class VectorStore:\n",
    "    def __init__(self, embed_model: str = \"llama3.2\"):\n",
    "        self.documents = []\n",
    "        self.embeddings = []\n",
    "        self.embed_model = embed_model\n",
    "        \n",
    "    async def get_embedding(self, text: str) -> List[float]:\n",
    "        async with httpx.AsyncClient() as client:\n",
    "            resp = await client.post(\n",
    "                \"http://localhost:11434/api/embeddings\",\n",
    "                json={\"model\": self.embed_model, \"prompt\": text}\n",
    "            )\n",
    "            return resp.json()[\"embedding\"]\n",
    "            \n",
    "    async def add_documents(self, docs: List[str]):\n",
    "        print(f\"Embedding {len(docs)} documents locally via Ollama...\")\n",
    "        self.documents.extend(docs)\n",
    "        for doc in docs:\n",
    "            emb = await self.get_embedding(doc)\n",
    "            self.embeddings.append(emb)\n",
    "        self.embeddings_np = np.array(self.embeddings)\n",
    "        print(\"Vector DB ready!\")\n",
    "\n",
    "    async def search(self, query: str, top_k: int = 2) -> List[str]:\n",
    "        # Embed the query\n",
    "        q_emb = await self.get_embedding(query)\n",
    "        q_np = np.array(q_emb)\n",
    "        \n",
    "        # Compute Cosine Similarity\n",
    "        norms = np.linalg.norm(self.embeddings_np, axis=1) * np.linalg.norm(q_np)\n",
    "        similarities = np.dot(self.embeddings_np, q_np) / norms\n",
    "        \n",
    "        # Get top K indices\n",
    "        top_indices = np.argsort(similarities)[::-1][:top_k]\n",
    "        return [self.documents[i] for i in top_indices]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Populate the Database with Company Documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "knowledge_base = [\n",
    "    \"Acme Corp standard shipping takes 3-5 business days. Expedited shipping is available for an extra $15 and takes 1-2 business days.\",\n",
    "    \"To reset your Acme Corp password, visit acmecorp.com/reset, enter your email, and click the link sent to your inbox. The link expires in 24 hours.\",\n",
    "    \"Acme Corp offers a 30-day money-back guarantee on all unopened electronics. Software licenses are non-refundable once activated.\",\n",
    "    \"For customer support, you can call 1-800-ACME-HELP between 9 AM and 5 PM EST, Monday through Friday, or email support@acmecorp.com.\",\n",
    "    \"Acme Corp's loyalty program gives you 1 point for every $10 spent. 100 points can be redeemed for a $5 discount.\"\n",
    "]\n",
    "\n",
    "vector_db = VectorStore(embed_model=\"llama3.2\")\n",
    "await vector_db.add_documents(knowledge_base)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Create the RAG Agent\n",
    "The agent will intercept the `TestCase`, pull exactly the right vectors from the `VectorStore`, use the LLM to generate an answer, and log the retrieved context for evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "class AdvancedRAGAgent:\n",
    "    def __init__(self, db: VectorStore, llm_provider):\n",
    "        self.db = db\n",
    "        self.llm = llm_provider\n",
    "\n",
    "    async def generate_answer(self, test_case: TestCase) -> str:\n",
    "        question = test_case.input\n",
    "        \n",
    "        # 1. RETRIEVE from Vector DB\n",
    "        retrieved_docs = await self.db.search(question, top_k=2)\n",
    "        \n",
    "        # 2. LOG CONTEXT into TestCase (for ContextPrecision/Recall metrics)\n",
    "        from mutant.eval.types import RAGContext\n",
    "        if test_case.rag is None:\n",
    "            test_case.rag = RAGContext()\n",
    "        test_case.rag.retrieval_context = retrieved_docs\n",
    "        \n",
    "        # 3. GENERATE answer using Ollama\n",
    "        context_str = \"\\n- \".join(retrieved_docs)\n",
    "        prompt = f\"\"\"You are a helpful customer support agent for Acme Corp.\n",
    "Answer the user's question using ONLY the provided context.\n",
    "\n",
    "CONTEXT:\n",
    "- {context_str}\n",
    "\n",
    "QUESTION:\n",
    "{question}\n",
    "\"\"\"\n",
    "        from mutant.providers import LLMMessage\n",
    "        messages = [LLMMessage(role=\"user\", content=prompt)]\n",
    "        response = await self.llm.complete(messages, temperature=0.1)\n",
    "        return response.content\n",
    "\n",
    "rag_agent = AdvancedRAGAgent(vector_db, provider)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Create the Evaluation Dataset\n",
    "Notice we provide `expected_output` (for relevancy/correctness) and `rag.context` (the **Golden Context** the DB *should* retrieve)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "from mutant.eval.types import RAGContext\n",
    "\n",
    "eval_dataset = [\n",
    "    TestCase(\n",
    "        input=\"How long does standard shipping take?\",\n",
    "        expected_output=\"Standard shipping takes 3-5 business days.\",\n",
    "        rag=RAGContext(\n",
    "            context=[\"Acme Corp standard shipping takes 3-5 business days. Expedited shipping is available for an extra $15 and takes 1-2 business days.\"]\n",
    "        )\n",
    "    ),\n",
    "    TestCase(\n",
    "        input=\"Can I get a refund on software?\",\n",
    "        expected_output=\"Software licenses are non-refundable once activated.\",\n",
    "        rag=RAGContext(\n",
    "            context=[\"Acme Corp offers a 30-day money-back guarantee on all unopened electronics. Software licenses are non-refundable once activated.\"]\n",
    "        )\n",
    "    )\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Run the End-to-End Evaluation\n",
    "Run the agent over the dataset, collect actual outputs & retrieval contexts, and let Mutant's Judges evaluate the quality!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "async def run_evaluation():\n",
    "    print(\"\\n🚀 Running RAG Agent on dataset...\\n\")\n",
    "    \n",
    "    # 1. Run the target\n",
    "    for tc in eval_dataset:\n",
    "        tc.actual_output = await rag_agent.generate_answer(tc)\n",
    "        \n",
    "    # 2. Setup the Evaluator Suite\n",
    "    metrics = [\n",
    "        Faithfulness(provider),       # Is the answer hallucinated?\n",
    "        AnswerRelevancy(provider),    # Did it answer the user's specific question?\n",
    "        ContextPrecision(provider),   # Did the VectorDB retrieve ONLY relevant context?\n",
    "        ContextRecall(provider)       # Did the VectorDB retrieve ALL golden context?\n",
    "    ]\n",
    "    \n",
    "    suite = EvalSuite(metrics=metrics, concurrency=2) # Concurrency 2 is safe for local Ollama\n",
    "    \n",
    "    print(\"⚖️ Evaluating outputs... (Using Ollama Locally!)\")\n",
    "    report = await suite.run(eval_dataset)\n",
    "    \n",
    "    # 3. Print CLI Report\n",
    "    report.display()\n",
    "    \n",
    "    # 4. Generate the HTML Dashboard!\n",
    "    report.to_html(\"ollama_rag_dashboard.html\")\n",
    "    print(\"\\n📊 Full HTML Dashboard saved to: ollama_rag_dashboard.html\")\n",
    "    \n",
    "await run_evaluation()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("advanced_rag_vector_db_showcase.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)
print("Notebook updated for Ollama!")
