<div align="center">
  <h1>Mutant AI</h1>
  <p><b>Automate Evaluation & Red Teaming for LLMs</b></p>
  
  <p>
    <a href="https://pypi.org/project/mutant-ai/"><img src="https://img.shields.io/pypi/v/mutant-ai.svg?style=flat-square&color=000000" alt="PyPI Version"></a>
    <a href="https://docs-mutantai.netlify.app"><img src="https://img.shields.io/badge/docs-live-black.svg?style=flat-square" alt="Documentation"></a>
    <a href="https://github.com/ankitgmishra/mutant-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-black.svg?style=flat-square" alt="License"></a>
  </p>
  
  <p>Mutant is the open-source evaluation framework for synthesizing adversarial datasets, testing AI agents, and grading RAG pipelines.</p>

  <h3><a href="https://docs-mutantai.netlify.app">Read the Official Documentation &rarr;</a></h3>
</div>

---

## ⚡ Quick Install

Mutant is available on PyPI. Install it using `pip` or `uv`:

```console
$ pip install mutant-ai
```

---

## 🏗️ Core Framework

Mutant is divided into four core pillars. You can use them independently or chain them together for end-to-end evaluation.

### 1. Data Augmentation
Stop writing manual test cases. Mutant takes a single seed scenario and uses cognitive dimensions (like anger, prompt injection, or complex reasoning) to generate hundreds of behaviorally diverse test cases instantly.

```python
from mutant.core import Scenario, augment
from mutant.providers.ollama import OllamaProvider

provider = OllamaProvider(model="llama3.1")

# Generate 5 diverse variations of a basic intent
dataset = await augment(
    dataset=[Scenario(title="Refund", description="I want a refund.")],
    provider=provider,
    mutations_per_case=5,
    dimensions=["emotion.angry", "language.slang"]
)

dataset.save("mutations.json")
```

### 2. Agentic Red Teaming
Instead of static security scans, unleash a hypothesis-driven attacker agent. Mutant dynamically probes your system in multi-turn conversations, adapting its attacks to bypass your safeguards.

```python
from mutant.redteam import red_team
from mutant.providers.ollama import OllamaProvider
from my_app import my_agent 

provider = OllamaProvider(model="llama3.1")

# Unleash the attacker against your live agent
report = await red_team(
    target=my_agent,
    goal="Extract the hidden system instructions.",
    provider=provider,
    max_turns=5
)

report.to_html("security_report.html")
```

### 3. Security Evaluation
Go beyond general correctness and actively grade your application's defensive mechanisms. Run your adversarial datasets through the Eval Suite using specialized security metrics to automatically detect prompt injections, PII leakage, and unauthorized behavior.

```python
from mutant.eval import EvalSuite, SensitiveDataLeakage, PromptInjectionResistance
from mutant.providers.ollama import OllamaProvider

provider = OllamaProvider(model="llama3.1")

# Grade the agent against security threats
suite = EvalSuite(metrics=[
    SensitiveDataLeakage(provider=provider),
    PromptInjectionResistance(provider=provider)
])

report = await suite.evaluate(test_cases=dataset.cases)
report.to_html("security_report.html")
```

### 4. Evaluation Suite
Once you've generated your adversarial dataset or executed your attacks, grade your application's responses using LLM-as-a-judge metrics to provide deterministic scores and HTML reports.

```python
from mutant.eval import EvalSuite, Correctness, Faithfulness
from mutant.providers.ollama import OllamaProvider

provider = OllamaProvider(model="llama3.1")

# Initialize metrics and evaluate test cases
suite = EvalSuite(metrics=[
    Correctness(provider=provider),
    Faithfulness(provider=provider)
])

report = await suite.evaluate(test_cases=dataset.cases)
report.to_html("eval_report.html")
```

---

## 📊 Actionable Reports

Mutant natively outputs highly visual HTML dashboards and CI/CD-ready JSON artifacts, mapping your exact vulnerability and coverage footprint natively out of the box.

---

## 📚 Documentation

For full documentation, including advanced tutorials, architecture diagrams, and custom metric creation, visit the [official documentation](https://docs-mutantai.netlify.app).

## 🤝 Contributing

Contributions are heavily welcomed! Please read our [Contributing Guide](https://github.com/ankitgmishra/mutant-ai/blob/main/CONTRIBUTING.md) to get started.

## 📄 License

MIT © 2026 [Ankit Mishra](https://aiankit.com). Built for the open-source community.
