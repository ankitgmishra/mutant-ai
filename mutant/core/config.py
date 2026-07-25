"""mutant/core/config.py — V0.5 Configuration object."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MutationConfig(BaseModel):
    """Configuration object for the mutation pipeline."""

    count: int = Field(50, description="Target number of mutations to generate.")
    concurrency: int = Field(5, description="Max concurrent LLM calls.")
    temperature: float = Field(0.8, description="Sampling temperature for the LLM.")
    max_tokens: int = Field(4096, description="Max tokens for completion.")

    # Stages
    quality_review: bool = Field(True, description="Enable quality review stage.")
    selective_quality_review_rate: float = Field(
        0.4, description="Fraction of cases to review if quality_review is enabled."
    )
    quality_batch_size: int = Field(10, description="Cases per quality review batch.")
    deduplicate: bool = Field(True, description="Enable semantic deduplication.")
    # Generation toggles
    generate_rationale: bool = Field(
        True,
        description="Whether the LLM should generate a rationale for its mutation.",
    )
    generate_tags: bool = Field(
        True, description="Whether the LLM should generate behavioral tags."
    )
    # Filters
    dimension_ids: list[str] | None = Field(
        None, description="Whitelist of dimension IDs to use."
    )
    exclude_dimension_ids: list[str] | None = Field(
        None, description="Blacklist of dimension IDs."
    )
    categories: list[str] | None = Field(
        None, description="Whitelist of MutationCategories."
    )
    severities: list[str] | None = Field(
        None, description="Whitelist of MutationSeverities."
    )

    # Developer Experience
    verbose: bool = Field(False, description="Enable progress reporting.")
    debug: bool = Field(
        False, description="Enable rich debug logging and trace exposure."
    )
    prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Optional prompt overrides (keys: 'behavior_analysis', 'mutation_planning', 'mutation_generation', 'quality_review', 'coverage_analysis').",
    )

    model_config = {"frozen": True}
