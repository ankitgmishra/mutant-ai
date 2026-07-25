# Changelog

All notable changes to Mutant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet.

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
