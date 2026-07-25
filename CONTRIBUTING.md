# Contributing to Mutant

Thank you for your interest in contributing! Mutant is a community project, and all contributions are welcome — from bug fixes and new mutation dimensions to documentation and examples.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Mutation Dimension](#adding-a-new-mutation-dimension)
- [Adding a New LLM Provider](#adding-a-new-llm-provider)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

---

## Code of Conduct

Please be respectful and constructive. This is a welcoming space for developers of all levels. Harassment, discrimination, or abusive behaviour will not be tolerated.

---

## How to Contribute

1. **Report a bug** — Open an [issue](https://github.com/ankitmishralive/mutant/issues) with a minimal reproduction.
2. **Suggest a feature** — Open an issue with `[Feature Request]` in the title and explain your use case.
3. **Fix a bug** — Comment on an open issue, fork the repo, and submit a pull request.
4. **Add a mutation dimension** — New dimensions are the lifeblood of Mutant. See the guide below.
5. **Improve docs** — Even fixing a typo is valuable.

---

## Development Setup

```bash
# 1. Fork and clone the repo
git clone https://github.com/<your-username>/mutant
cd mutant

# 2. Create a virtual environment (uv recommended)
uv venv
source .venv/bin/activate

# 3. Install all dev dependencies
uv pip install -e ".[dev,gemini,openai]"

# 4. Install pre-commit hooks (auto-formats and lints on every commit)
pre-commit install

# 5. Verify setup
pytest
```

---

## Project Structure

```
mutant/
├── core/           # Engine, data models, config, registry
├── pipeline/       # 5-stage async pipeline + prompt templates
├── dimensions/     # Built-in mutation dimensions (add yours here!)
├── providers/      # LLM adapters (Ollama, Gemini, OpenAI, Anthropic, LiteLLM)
├── coverage/       # Coverage analysis + HTML report
├── datasets/       # Dataset I/O utilities (load_csv, load_json)
├── reports/        # Export formats
└── cache/          # Disk-based LLM response cache
tests/              # Pytest test suite
examples/           # Runnable usage examples
docs/               # MkDocs documentation source
```

---

## Adding a New Mutation Dimension

Dimensions live in `mutant/dimensions/`. Each file corresponds to a category (e.g., `safety.py`, `emotion.py`).

### Step 1 — Create your dimension class

```python
# In the appropriate file, e.g., mutant/dimensions/safety.py

from mutant.dimensions.base import MutationDimension
from mutant.core.mutation import MutationCategory, MutationSeverity

class MyNewAttack(MutationDimension):
    id = "safety.my_new_attack"
    name = "My New Attack"
    description = "Brief description of what this attack tests."
    category = MutationCategory.SAFETY
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Rewrite the user message to embed [describe the attack]. "
            "The rewrite must still seem like a genuine user request."
        )

    def get_examples(self) -> list[str]:
        return [
            "Example 1 of the mutation in action.",
            "Example 2 of the mutation in action.",
        ]
```

### Step 2 — Register it

Open `mutant/dimensions/__init__.py` and add your class to the `_ALL_DIMENSIONS` list.

### Step 3 — Write a test

Add a test in `tests/` verifying the dimension is registered and has valid fields.

### Step 4 — Open a PR

---

## Adding a New LLM Provider

Providers live in `mutant/providers/`. Each provider subclasses `BaseLLMProvider`.

```python
from mutant.providers.base import BaseLLMProvider, LLMMessage, LLMResponse

class MyProvider(BaseLLMProvider):
    provider_name = "myprovider"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._api_key = api_key

    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # Call your API here
        ...
        return LLMResponse(content=raw_text, model=self.model, metadata={})
```

The base class handles JSON parsing, retry logic, and schema validation automatically.

---

## Testing

```bash
# Run all tests
pytest

# Run a specific file
pytest tests/test_mutation.py -v

# Run with coverage report
pytest --cov=mutant --cov-report=html
```

All new code should come with tests. We target **≥ 75% coverage**.

---

## Pull Request Process

1. **Branch naming**: `feat/<feature>`, `fix/<bug>`, `docs/<topic>`
2. **Commit messages**: Use [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `chore:`
3. **PR description**: Explain *what* changed and *why*. Reference any related issues.
4. **CI must pass**: All tests, linting (`ruff`), and type-checks (`mypy`) must be green.
5. **One reviewer approval** required before merge.

---

## Style Guide

- **Formatter**: `ruff format` (enforced by pre-commit)
- **Linter**: `ruff check` (enforced by pre-commit)
- **Type checker**: `mypy --strict`
- **Docstrings**: Google-style for public APIs
- **Line length**: 88 characters

Run all checks manually:

```bash
ruff format .
ruff check .
mypy mutant/
```

---

Thank you for making Mutant better! 🧬
