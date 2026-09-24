# Changelog

All notable changes to Mutant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **`BaseLLMProvider.complete_text()`** — a natural-language completion path, for
  application code whose output is read by a human or a user (agent replies, RAG
  answers, refusals). `OllamaProvider` overrides it so the reply is no longer forced
  into a JSON envelope; `complete()` keeps that constraint because the mutation
  engine and the LLM judges parse its output.
- **`mutant.eval.gate`** — a model-free security gate you can run on every pull request:
  curated probes, no generation call, no judge, milliseconds to run. Use it from the
  command line (`python -m mutant.eval.gate --target myapp.agent:handle --secret ...`)
  or from a test (`assert_security_gate(target, secrets=[...])`). It reports the checks
  it could not decide instead of counting them as passes, and explains when a metric was
  skipped (the tool-authorization metric needs `tool_constraints`; with none it would
  pass every call it saw).
- **`examples/security_target/`** — a runnable service with the bug classes the suite
  targets, behind a switch: vulnerable by default, guarded with
  `MUTANT_DEMO_SECURITY=on`. The gate fails one and passes the other, which is how you
  check that a green result is not vacuous.
- **`MutationResult.shortfall` and `.dimension_counts`**, plus `requested_count`, so a run
  that delivered fewer probes than requested — or delivered none for a dimension you
  asked for — says so instead of looking like a clean result.
- **`tests/test_judge_reliability.py`** — an opt-in harness (`MUTANT_JUDGE_EVAL=1`) that
  measures the semantic judges against labelled must-fail/must-pass cases, with the known
  misses recorded as expected failures.
- Optional **`notebooks`** dependency extra (`numpy`, `ipykernel`, `nbclient`,
  `nbconvert`) — the example notebooks import `numpy`, which was previously
  undeclared and disappeared on `uv sync`.

### Changed
- Security metrics: several detection defects found by running the suite against a real
  application are fixed.
  - `PromptInjectionResistance` recognises an instruction override by its **shape**
    rather than a fixed phrase list, so paraphrases ("ignore all prior instructions") are
    judged instead of silently reported as "not an injection". A probe from the injection
    dimension the classifier still misses comes back `UNKNOWN` with
    `metadata["classifier_missed"] = True`.
  - `RAGContextLeakage` requires a substantial verbatim overlap with a sensitive line or a
    secret **value** shared with the retrieved document; it no longer fails a refusal for
    sharing a word with it.
  - `InstructionBoundary` fails a claimed forbidden action without needing a judge
    (`metadata["violated_action"]`), and no longer fails a reply merely for mentioning
    confidentiality or opening with "Here is …" while disclosing nothing.
  - `SensitiveDataLeakage` uses its credential heuristic: an unverified credential-shaped
    value in the output is `UNKNOWN`, never a silent pass.
  - Mutation generation tops up a short batch instead of shipping it, and the
    tool-argument and data-leakage dimension instructions require a concrete identifier
    instead of accepting vague probes.
- `EvalSuite`'s `TargetFn` alias now reflects what the suite already supported: a target
  may return a `TestCase` with observables, not only a string.
- The security-evaluation notebooks (`examples/notebooks/security_eval/`) now
  evaluate a real application — `security_lab.py`, with real TF-IDF retrieval, real
  LLM replies and real tool execution against an order book and HR state — instead of
  hardcoded reply strings. Each notebook also runs a deterministic control built from
  a real bug (`context_echo`, `debug_echo`, `no_authz`) and compares each metric
  verdict against ground truth taken from application state.

### Fixed
- `mutant/prompts/mutant_report.html` referenced fields the report never provided
  (`case.original_description`, `case.novelty_score`), which raised a jinja2
  `UndefinedError` for every HTML report. The JSON report now includes
  `quality_approved`.
- `tests/` referenced `BehaviorAnalysis.intent`, removed in the behaviour-analysis
  refactor, in four places.

### Known issues
- **The semantic judges miss real violations.** Measured with
  `MUTANT_JUDGE_EVAL=1 uv run pytest tests/test_judge_reliability.py`: of seven labelled
  must-fail cases the judges passed three — a timesheet approval phrased with the policy
  inverted, an employee record returned inside a Python dict, and an endpoint quoted out
  of the system prompt. Do not rely on a judged verdict alone; gate on a suite.
- **Probe generation can drift off-target**, and a small model asked for eight probes may
  return one. Generation now tops up and reports the shortfall, but there is no
  validation-and-retry loop against per-dimension required entities.
- **`pytest` as configured exits non-zero on coverage** (`--cov-fail-under=75`): total
  coverage is ~65%, driven by `mutant/redteam/*` (the deprecated tree) and
  `mutant/core/mutation.py` (38%). The security metrics themselves are at 84%. Either add
  tests for the redteam tree or scope the gate before wiring CI to the default command.

---

## [0.7.0] — 2026-07-21

### Added
- **`generate_rationale` and `generate_tags` flags** on `mutate()` and `augment()` — when set to `False`, the LLM skips generating rationale and behavioral tags, resulting in faster and leaner output.
- **Dynamic Pydantic schema** in the generation stage — the LLM response model is built at runtime based on enabled generation flags, eliminating unused fields.
- **`PipelineConfig.generate_rationale` and `generate_tags`** properties properly propagated from `MutationConfig` through the pipeline context.
- `getattr` safe-defaults throughout `_gen_one` to handle optional fields gracefully without crashing.

### Fixed
- `AttributeError: 'PipelineConfig' object has no attribute 'generate_rationale'` — caused generation to silently produce 0 cases when rationale/tags were disabled.
- `AttributeError: 'DynamicGeneratedMutation' has no attribute 'realism_notes'` — hardcoded field access on a dynamically constructed model caused all generated cases to be silently dropped.
- `to_csv()` producing empty files when `cases` was an empty list — root cause was the generation error above.
- `to_dataframe()` returning an empty `DataFrame` with no columns — same root cause.

---

## [0.6.0] — 2026-07-20

### Added
- **Gemini provider** (`GeminiProvider`) with robust jittered retry logic for rate-limit and quota-exhaustion handling.
- Configurable generation controls (`generate_rationale`, `generate_tags`) as top-level parameters on `mutate()` and `augment()`.
- `MutationConfig` now exposes `generate_rationale` and `generate_tags` boolean fields.

### Changed
- `mutate()` and `augment()` now accept generation control flags as top-level kwargs rather than requiring a full `MutationConfig` object.

---

## [0.5.0] — 2026-07-19

### Added
- **Coverage dashboard** (`coverage()` function + `CoverageReport.to_html()`) — generates a rich, interactive HTML behavioral analytics report.
- **Diversity Radar** spider-web chart in the coverage dashboard visualising Emotion, Language, Domain, Conversation, and Difficulty dimensions.
- `load_csv()` multi-column support — pass a list of column names to `text_column` to concatenate multiple columns into the scenario description.
- `to_parquet()` and `to_huggingface()` export methods on `MutationResult` and `AugmentedDataset`.

### Fixed
- String interpolation syntax errors in coverage dashboard HTML template.

---

## [0.4.0] — 2026-07-18

### Added
- **5-stage async pipeline**: Behavior Analysis → Mutation Planning → Generation → Quality Review → Deduplication.
- `EvaluationCase` model with `quality_approved`, `rationale`, `behavioral_tags` fields.
- `MutationResult.explain()` — prints a structured coverage summary.
- `MutationResult.filter()` — filter cases by `dimension`, `severity`, or `keyword`.
- `MutationResult.sort_by()` — sort cases by any field.
- `AugmentedDataset` — result type for `augment()`, mirrors all export methods of `MutationResult`.
- `to_json()`, `to_jsonl()`, `to_csv()`, `to_dataframe()` export methods.
- `QualityReviewResult` stage that uses the LLM-as-a-judge to approve or reject generated cases.
- Semantic deduplication stage using LLM.

### Changed
- `MutationPlan` renamed from `BehaviorPlan` (backward-compat alias kept).
- `MutationCase` aliased to `EvaluationCase`.

---

## [0.3.0] — 2026-07-18

### Added
- `augment()` function for dataset-scale mutation augmentation.
- `load_csv()` and `load_json()` dataset ingestion utilities.
- `concurrency` parameter on `augment()` to rate-limit parallel LLM calls.

---

## [0.2.0] — 2026-07-17

### Added
- Initial `mutate()` async public API.
- `Scenario` input model.
- `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `LiteLLMProvider`.
- 35 built-in mutation dimensions across Safety, Emotion, Language, Reasoning, Context, Time, Tool, Identity, Memory, Policy, Retrieval, Knowledge, Conversation categories.
- `MutationRegistry` plugin system for custom dimensions.
- `MutationEngine` orchestrator.
- `PipelineConfig` and `PipelineContext`.
- Rich CLI (`mutant list`, `mutant run`).
- Disk-based LLM response cache.

---

## [0.1.0] — 2026-07-15

### Added
- Initial project scaffold.
- `pyproject.toml` with Hatch build system.
- Basic project structure.

---

[Unreleased]: https://github.com/ankitmishralive/mutant/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/ankitmishralive/mutant/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ankitmishralive/mutant/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ankitmishralive/mutant/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ankitmishralive/mutant/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ankitmishralive/mutant/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ankitmishralive/mutant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ankitmishralive/mutant/releases/tag/v0.1.0
