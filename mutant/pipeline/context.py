"""mutant/pipeline/context.py — V0.4"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mutant.core.mutation import (
    BehaviorAnalysis,
    EvaluationCase,
    MutationPlan,
    QualityReviewResult,
)
from mutant.core.scenario import Scenario


class PipelineConfig(BaseModel):
    count: int = 50
    concurrency: int = 5
    temperature: float = 0.8
    max_tokens: int = 4096
    quality_review: bool = True
    quality_batch_size: int = 10
    deduplicate: bool = True
    generate_rationale: bool = True
    generate_tags: bool = True
    dimension_ids: list[str] | None = None
    category_filter: list[str] | None = None
    severity_filter: list[str] | None = None
    prompts: dict[str, str] = Field(default_factory=dict)
    model_config = {"frozen": True}


class PipelineContext(BaseModel):
    scenario: Scenario
    config: PipelineConfig

    # Stage outputs
    behavior_analysis: BehaviorAnalysis | None = None
    mutation_plan: MutationPlan | None = None
    raw_cases: list[EvaluationCase] = Field(default_factory=list)
    reviewed_cases: list[EvaluationCase] = Field(default_factory=list)
    quality_review_result: QualityReviewResult | None = None
    deduplicated_cases: list[EvaluationCase] = Field(default_factory=list)
    final_cases: list[EvaluationCase] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def output_cases(self) -> list[EvaluationCase]:
        return (
            self.final_cases
            or self.deduplicated_cases
            or self.reviewed_cases
            or self.raw_cases
        )
