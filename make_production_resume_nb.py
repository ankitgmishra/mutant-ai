import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mutant Production-Level RAG Evaluation (LLM-as-a-Judge)\n",
    "\n",
    "This notebook demonstrates a **True, Production-Grade RAG Pipeline** running entirely locally. \n",
    "It reads your actual resume (`Ankit Mishra 2026.pdf`), processes it, indexes it into a local Numpy-based TF-IDF Vector Database, and evaluates the RAG Agent's ability to answer recruiter questions accurately.\n",
    "\n",
    "Finally, it uses Mutant's **LLM-as-a-Judge** metrics (`Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`) to score the pipeline and outputs a beautiful HTML dashboard."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import asyncio\n",
    "import numpy as np\n",
    "import re\n",
    "from typing import List\n",
    "\n",
    "# Mutant Imports\n",
    "from mutant.providers import OllamaProvider\n",
    "from mutant.eval.types import TestCase, RAGContext\n",
    "from mutant.eval.suite import EvalSuite\n",
    "from mutant.eval.metrics import (\n",
    "    Faithfulness,\n",
    "    AnswerRelevancy,\n",
    "    ContextPrecision,\n",
    "    ContextRecall\n",
    ")\n",
    "\n",
    "# Set up the LLM Provider for generation & evaluation\n",
    "provider = OllamaProvider(model=\"llama3.2\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Build the Vector Database & Ingest the Resume\n",
    "Since your local Ollama server doesn't have the `--embeddings` flag enabled, we built a mathematically robust TF-IDF Vectorizer using pure Numpy! This requires zero API calls and is lightning fast."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "class TFIDFVectorStore:\n",
    "    def __init__(self):\n",
    "        self.documents = []\n",
    "        self.vocab = {}\n",
    "        self.idf = None\n",
    "        self.tf_idf_matrix = None\n",
    "        \n",
    "    def _tokenize(self, text: str) -> List[str]:\n",
    "        return re.findall(r'\\b\\w+\\b', text.lower())\n",
    "        \n",
    "    async def add_documents(self, docs: List[str]):\n",
    "        print(f\"Indexing {len(docs)} chunks of your Resume via Numpy TF-IDF...\")\n",
    "        self.documents.extend(docs)\n",
    "        \n",
    "        # Build vocabulary\n",
    "        doc_words = [self._tokenize(d) for d in docs]\n",
    "        words = set([w for doc in doc_words for w in doc])\n",
    "        self.vocab = {w: i for i, w in enumerate(words)}\n",
    "        \n",
    "        # Compute Term Frequency (TF)\n",
    "        tf = np.zeros((len(docs), len(self.vocab)))\n",
    "        for i, doc in enumerate(doc_words):\n",
    "            for w in doc:\n",
    "                tf[i, self.vocab[w]] += 1\n",
    "                \n",
    "        # Compute Inverse Document Frequency (IDF)\n",
    "        df = np.sum(tf > 0, axis=0)\n",
    "        self.idf = np.log(len(docs) / (df + 1))\n",
    "        \n",
    "        # TF-IDF matrix\n",
    "        self.tf_idf_matrix = tf * self.idf\n",
    "        \n",
    "        # Normalize the vectors (L2 norm)\n",
    "        norms = np.linalg.norm(self.tf_idf_matrix, axis=1, keepdims=True)\n",
    "        norms[norms == 0] = 1\n",
    "        self.tf_idf_matrix = self.tf_idf_matrix / norms\n",
    "        print(\"\\n✅ Vector DB Ready!\")\n",
    "\n",
    "    async def search(self, query: str, top_k: int = 3) -> List[str]:\n",
    "        q_words = self._tokenize(query)\n",
    "        q_vec = np.zeros(len(self.vocab))\n",
    "        for w in q_words:\n",
    "            if w in self.vocab:\n",
    "                q_vec[self.vocab[w]] += 1\n",
    "                \n",
    "        q_vec = q_vec * self.idf\n",
    "        norm = np.linalg.norm(q_vec)\n",
    "        if norm > 0:\n",
    "            q_vec = q_vec / norm\n",
    "            \n",
    "        similarities = np.dot(self.tf_idf_matrix, q_vec)\n",
    "        top_indices = np.argsort(similarities)[::-1][:top_k]\n",
    "        return [self.documents[i] for i in top_indices]\n",
    "\n",
    "# --------------------------------------------------------\n",
    "# Load and Chunk Resume\n",
    "# --------------------------------------------------------\n",
    "with open(\"resume_text.txt\", \"r\") as f:\n",
    "    resume_text = f.read()\n",
    "\n",
    "# Simple chunking by double newlines (paragraphs/sections)\n",
    "raw_chunks = [c.strip() for c in resume_text.split(\"\\n\\n\") if len(c.strip()) > 50]\n",
    "\n",
    "vector_db = TFIDFVectorStore()\n",
    "await vector_db.add_documents(raw_chunks)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. The Agent\n",
    "The RAG Agent searches your resume and answers recruiter questions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "class ResumeAgent:\n",
    "    def __init__(self, db: TFIDFVectorStore, llm_provider):\n",
    "        self.db = db\n",
    "        self.llm = llm_provider\n",
    "\n",
    "    async def generate_answer(self, test_case: TestCase) -> str:\n",
    "        # 1. Retrieve the most relevant resume sections\n",
    "        retrieved_docs = await self.db.search(test_case.input, top_k=3)\n",
    "        \n",
    "        # 2. Log context for Mutant Evaluation\n",
    "        if test_case.rag is None:\n",
    "            test_case.rag = RAGContext()\n",
    "        test_case.rag.retrieval_context = retrieved_docs\n",
    "        \n",
    "        # 3. Generate Answer\n",
    "        context_str = \"\\n\\n---\\n\\n\".join(retrieved_docs)\n",
    "        prompt = f\"\"\"You are an AI assistant representing Ankit Mishra. \n",
    "You are talking to a recruiter. Answer their question professionally, using ONLY the provided resume context.\n",
    "If the answer isn't in the context, say you don't know.\n",
    "\n",
    "RESUME CONTEXT:\n",
    "{context_str}\n",
    "\n",
    "RECRUITER QUESTION:\n",
    "{test_case.input}\n",
    "\"\"\"\n",
    "        from mutant.providers import LLMMessage\n",
    "        messages = [LLMMessage(role=\"user\", content=prompt)]\n",
    "        response = await self.llm.complete(messages, temperature=0.2)\n",
    "        return response.content\n",
    "\n",
    "agent = ResumeAgent(vector_db, provider)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Production Golden Dataset\n",
    "Here we define what a *perfect* system should output and retrieve."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "eval_dataset = [\n",
    "    TestCase(\n",
    "        input=\"Where did Ankit study and what was his GPA?\",\n",
    "        expected_output=\"Ankit studied Artificial Intelligence & Machine Learning at the University of Mumbai, where he achieved a GPA of 3.5/4.\",\n",
    "        rag=RAGContext(\n",
    "            context=[\"University of Mumbai\\nBachelor of Technology in Artificial Intelligence & Machine Learning\\nGPA: 3.5 / 4\"]\n",
    "        )\n",
    "    ),\n",
    "    TestCase(\n",
    "        input=\"What experience does he have with DocuraHealth?\",\n",
    "        expected_output=\"He worked at DocuraHealth (YC W26) in San Francisco.\",\n",
    "        rag=RAGContext(\n",
    "            context=[\"WORK EXPERIENCE\\nDocuraHealth (YC W26)\\nSan Francisco, USA\"]\n",
    "        )\n",
    "    )\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Evaluate using LLM-as-a-Judge\n",
    "Run the local LLM judges on the RAG pipeline!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "async def run_evaluation():\n",
    "    print(\"\\n🚀 Running Agent on the Golden Dataset...\\n\")\n",
    "    for tc in eval_dataset:\n",
    "        tc.actual_output = await agent.generate_answer(tc)\n",
    "        \n",
    "    # Initialize Judges\n",
    "    metrics = [\n",
    "        Faithfulness(provider),\n",
    "        AnswerRelevancy(provider),\n",
    "        ContextPrecision(provider),\n",
    "        ContextRecall(provider)\n",
    "    ]\n",
    "    \n",
    "    # Throttled concurrency so your local GPU doesn't crash\n",
    "    suite = EvalSuite(metrics=metrics, concurrency=1)\n",
    "    \n",
    "    print(\"⚖️ Evaluating... (This may take a few minutes locally)\")\n",
    "    report = await suite.run(eval_dataset)\n",
    "    \n",
    "    # Print the CLI table\n",
    "    report.display()\n",
    "    \n",
    "    # Save the Beautiful Dark-Themed Dashboard\n",
    "    report.to_html(\"production_resume_dashboard.html\")\n",
    "    print(\"\\n📊 Full HTML Dashboard saved to: production_resume_dashboard.html\")\n",
    "\n",
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

with open("production_resume_rag_showcase.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)
