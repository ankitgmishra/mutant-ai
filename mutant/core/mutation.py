"""
mutant/core/mutation.py  — V0.4
================================
All data models for the mutation pipeline.
No generation logic lives here.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Enumerations ───────────────────────────────────────────────────────────────


class MutationCategory(StrEnum):
    CONTEXT = "context"
    LANGUAGE = "language"
    EMOTION = "emotion"
    MEMORY = "memory"
    TIME = "time"
    TOOL = "tool"
    REASONING = "reasoning"
    SAFETY = "safety"
    INTENT = "intent"
    IDENTITY = "identity"
    POLICY = "policy"
    KNOWLEDGE = "knowledge"
    CONVERSATION = "conversation"
    RETRIEVAL = "retrieval"


class MutationSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Stage 1 output: Behavior Analysis ─────────────────────────────────────────


class BehaviorAnalysis(BaseModel):
    """Rich structural analysis of a scenario produced by the LLM."""

    detected_domain: str = Field(
        default="", description="Inferred domain (e.g. e-commerce, healthcare)."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Domain detection confidence."
    )
    actors: list[str] = Field(
        default_factory=list, description="People or systems involved."
    )
    entities: list[str] = Field(
        default_factory=list, description="Key objects, values, identifiers."
    )
    goals: list[str] = Field(
        default_factory=list, description="What actors are trying to achieve."
    )
    constraints: list[str] = Field(
        default_factory=list, description="Rules or limits that apply."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Things the agent implicitly assumes."
    )
    policies: list[str] = Field(
        default_factory=list, description="Business or domain policies in play."
    )
    tools: list[str] = Field(
        default_factory=list, description="Tools or APIs the agent would use."
    )
    risks: list[str] = Field(
        default_factory=list, description="Areas where failure is likely or costly."
    )
    likely_failure_modes: list[str] = Field(
        default_factory=list, description="How a weak agent would fail."
    )
    ambiguities: list[str] = Field(
        default_factory=list, description="Underspecified or unclear elements."
    )


# ── Stage 2 output: Mutation Plan ─────────────────────────────────────────────


class DimensionAllocation(BaseModel):
    """Planner decision for one mutation dimension."""

    dimension_id: str
    dimension_name: str
    count: int
    priority: int
    rationale: str
    why_selected: str = Field(
        default="",
        description="Explicit explanation of why this dimension matters here.",
    )
    focus_areas: list[str] = Field(default_factory=list)
    difficulty: str = Field(default="medium", description="low | medium | high")
    mutation_type: str = Field(default="single", description="single | composed")


class MutationPlan(BaseModel):
    """Planner output — exposed in MutationResult for debugging."""

    dimension_allocations: list[DimensionAllocation]
    relevant_dimensions: list[str] = Field(
        default_factory=list,
        description="IDs of dimensions applicable to this scenario",
    )
    irrelevant_dimensions: list[str] = Field(
        default_factory=list, description="IDs of dimensions not applicable"
    )
    coverage_strategy: str
    diversity_strategy: str = ""
    total_planned: int = 0
    expected_failure_modes: list[str] = Field(default_factory=list)


# Backward-compat alias
BehaviorPlan = MutationPlan


# ── Stage 3 LLM response: Generated Mutation ──────────────────────────────────


class GeneratedMutationMinimal(BaseModel):
    mutated_description: str


class GeneratedMutation(BaseModel):
    """Raw output from the mutation generation LLM call."""

    mutated_description: str
    rationale: str = ""
    behavioral_tags: list[str] = Field(default_factory=list)
    realism_notes: str = ""


class MutationPlanResponse(BaseModel):
    plans: list[MutationPlanItem]


class MutationPlanItem(BaseModel):
    plan_id: str
    title: str
    behavioral_challenge: str
    transformation_description: str
    key_elements: list[str] = Field(default_factory=list)
    avoid_elements: list[str] = Field(default_factory=list)


# ── Stage 4 output: Quality Review ────────────────────────────────────────────


class QualityScore(BaseModel):
    """LLM quality verdict for a single mutation case."""

    case_id: str
    preserves_task: bool
    preserves_domain: bool
    meaningful_mutation: bool
    realistic: bool
    sufficiently_different: bool
    approved: bool
    rejection_reason: str | None = None

    @property
    def overall_score(self) -> float:
        return 1.0 if self.approved else 0.0


class QualityReviewResult(BaseModel):
    """Output of the quality review stage."""

    scores: list[QualityScore] = Field(default_factory=list)
    approved_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)


# ── Stage 5 output: Deduplication ─────────────────────────────────────────────


class DeduplicationResult(BaseModel):
    duplicate_groups: list[dict[str, Any]] = Field(default_factory=list)
    unique_ids: list[str] = Field(default_factory=list)
    deduplication_summary: str = ""


# ── Primary output: EvaluationCase ────────────────────────────────────────────


class EvaluationCase(BaseModel):
    """A single mutation case — rich enough to be used directly in evaluation.

    Includes the mutated input, rationale, expected behaviors, and failure
    modes so it can be plugged directly into any evaluation framework.
    """

    id: str = Field(exclude=True)
    parent_id: str | None = Field(default=None, exclude=True)
    dimension_id: str
    dimension_name: str
    category: MutationCategory
    severity: MutationSeverity
    original_description: str
    mutated_description: str
    rationale: str = ""
    behavioral_tags: list[str] = Field(default_factory=list)
    # Scores
    quality_approved: bool = True

    model_config = {"frozen": True}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationCase(dim={self.dimension_name!r}, "
            f"sev={self.severity.value}, "
            f"approved={self.quality_approved})"
        )


# Backward-compat alias — existing code using MutationCase continues to work
MutationCase = EvaluationCase


# ── Top-level result ───────────────────────────────────────────────────────────


class MutationResult(BaseModel):
    """Complete output of a mutation run."""

    cases: list[EvaluationCase]
    behavior_analysis: BehaviorAnalysis | None = None
    mutation_plan: MutationPlan | None = None

    @property
    def count(self) -> int:
        return len(self.cases)

    @property
    def coverage_score(self) -> float:
        if not self.mutation_plan or not self.mutation_plan.relevant_dimensions:
            return 0.0
        explored = {c.dimension_id for c in self.cases}
        return len(explored) / max(len(self.mutation_plan.relevant_dimensions), 1)

    @property
    def summary(self) -> str:
        return f"MutationResult: {len(self.cases)} cases generated. Coverage: {self.coverage_score:.0%}."

    def filter(self, **kwargs: Any) -> MutationResult:
        """Filter cases based on attributes."""
        filtered = self.cases
        for key, value in kwargs.items():
            if key == "dimension":
                filtered = [
                    c
                    for c in filtered
                    if value.lower() in c.dimension_id.lower()
                    or value.lower() in c.dimension_name.lower()
                ]
            elif key == "severity":
                filtered = [
                    c for c in filtered if c.severity.value.lower() == value.lower()
                ]
            elif key == "keyword":
                keyword = value.lower()
                filtered = [
                    c
                    for c in filtered
                    if keyword in c.mutated_description.lower()
                    or keyword in c.rationale.lower()
                ]
            else:
                filtered = [c for c in filtered if getattr(c, key, None) == value]
        return self.model_copy(update={"cases": filtered})

    def explain(self, print_output: bool = True) -> None:
        """Print a structured explanation of the generation process and coverage."""
        if not print_output:
            return

        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        console = Console()
        console.print()

        header_text = Text.from_markup(
            "[bold cyan]MUTATION GENERATION EXPLANATION[/bold cyan]\n"
            "[dim]Planner Strategy and Coverage Summary[/dim]"
        )
        console.print(
            Panel(header_text, box=box.DOUBLE, border_style="cyan", padding=(1, 4)),
            justify="center"
        )
        console.print()

        if self.mutation_plan:
            console.rule("[bold cyan]PLANNER STRATEGY[/bold cyan]", style="cyan")
            console.print()
            console.print(f"  [dim]Strategy:[/dim]              [bold white]{self.mutation_plan.coverage_strategy}[/bold white]")
            
            rel = ', '.join(self.mutation_plan.relevant_dimensions) if self.mutation_plan.relevant_dimensions else 'None'
            console.print(f"  [dim]Relevant Dimensions:[/dim]   [white]{rel}[/white]")
            
            skip = ', '.join(self.mutation_plan.irrelevant_dimensions) if self.mutation_plan.irrelevant_dimensions else 'None'
            console.print(f"  [dim]Skipped Dimensions:[/dim]    [white]{skip}[/white]")
            console.print()

            explored = {c.dimension_id for c in self.cases}
            rel_list = self.mutation_plan.relevant_dimensions or []
            unexplored = [d for d in rel_list if d not in explored]
            
            console.rule("[bold cyan]COVERAGE SUMMARY[/bold cyan]", style="cyan")
            console.print()
            
            score_color = "green" if self.coverage_score == 1.0 else ("yellow" if self.coverage_score > 0.5 else "red")
            console.print(f"  [dim]Coverage Score:[/dim]        [{score_color} bold]{self.coverage_score:.0%}[/{score_color} bold]")
            console.print(f"  [dim]Explored:[/dim]              [white]{len(explored)} dimensions out of {len(rel_list)}[/white]")
            
            if unexplored:
                console.print()
                console.print("  [bold red]Gaps remain in:[/bold red]")
                for gap in unexplored:
                    console.print(f"    [red]•[/red] {gap}")
            
            console.print()

    # ── Export Methods ────────────────────────────────────────────────────────

    def _to_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Convert cases to a flat list of dicts suitable for tabular export."""
        # Default exclusions for cleaner output
        default_excludes = {
            "dimension_name",
            "category",
            "severity",
            "rationale",
            "behavioral_tags",
        }
        exclude = {k for k in default_excludes if kwargs.get(k) is not True}

        # Add explicit exclusions from kwargs
        for k, v in kwargs.items():
            if v is False and k != "mutated_description":
                exclude.add(k)

        records = []
        for case in self.cases:
            record = case.model_dump(exclude=exclude if exclude else None)
            if "category" in record and isinstance(record["category"], Enum):
                record["category"] = record["category"].value
            if "severity" in record and isinstance(record["severity"], Enum):
                record["severity"] = record["severity"].value
            # JSON-ify complex fields for flat tabular formats
            import json

            for key in ["behavioral_tags", "generation_metadata", "metadata"]:
                if record.get(key):
                    record[key] = json.dumps(record[key])
            records.append(record)
        return records

    def to_dataframe(self, **kwargs: Any) -> Any:
        """Convert results to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install it with `pip install pandas`"
            )
        return pd.DataFrame(self._to_records(**kwargs))

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Export results to CSV."""
        import csv

        records = self._to_records(**kwargs)
        if not records:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def sort_by(self, field: str, descending: bool = False) -> MutationResult:
        """Sort the cases in this result by a specific field."""
        if not self.cases:
            return self

        def _key(c: EvaluationCase) -> Any:
            val = getattr(c, field)
            if isinstance(val, Enum):
                # Simple heuristic for enums (like Severity) to sort by their order
                return list(type(val)).index(val)
            return val

        sorted_cases = sorted(self.cases, key=_key, reverse=descending)
        return self.model_copy(update={"cases": sorted_cases})

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Any:
        return iter(self.cases)

    def __getitem__(self, index: int | slice) -> Any:
        return self.cases[index]

    def to_json(self, path: str, **kwargs: Any) -> None:
        """Export full result object to JSON."""
        exclude = {
            k for k, v in kwargs.items() if v is False and k != "mutated_description"
        }
        exclude_dict = {"cases": {"__all__": exclude}} if exclude else None
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, exclude=exclude_dict))

    def to_jsonl(self, path: str, **kwargs: Any) -> None:
        """Export cases to a JSON Lines file."""
        exclude = {
            k for k, v in kwargs.items() if v is False and k != "mutated_description"
        }
        with open(path, "w", encoding="utf-8") as f:
            for case in self.cases:
                f.write(
                    case.model_dump_json(exclude=exclude if exclude else None) + "\n"
                )

    def to_parquet(self, path: str, **kwargs: Any) -> None:
        """Export results to Parquet."""
        df = self.to_dataframe(**kwargs)
        df.to_parquet(path)

    def to_huggingface(self, **kwargs: Any) -> Any:
        """Convert results to a HuggingFace Dataset."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets is required for to_huggingface(). Install it with `pip install datasets`"
            )
        return Dataset.from_list(self._to_records(**kwargs))


# ── AugmentedDataset ───────────────────────────────────────────────────────────


class AugmentedDataset(BaseModel):
    """Result of augmenting an entire dataset."""

    cases: list[EvaluationCase]

    @property
    def summary(self) -> str:
        return f"AugmentedDataset: {len(self.cases)} total cases generated."

    def _to_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        # Default exclusions for cleaner output
        default_excludes = {
            "dimension_name",
            "category",
            "severity",
            "rationale",
            "behavioral_tags",
            "expected_failure_modes",
        }
        exclude = {k for k in default_excludes if kwargs.get(k) is not True}

        for k, v in kwargs.items():
            if v is False and k != "mutated_description":
                exclude.add(k)

        records = []
        for case in self.cases:
            record = case.model_dump(exclude=exclude if exclude else None)
            if "category" in record and isinstance(record["category"], Enum):
                record["category"] = record["category"].value
            if "severity" in record and isinstance(record["severity"], Enum):
                record["severity"] = record["severity"].value
            import json

            for key in ["behavioral_tags", "expected_failure_modes"]:
                if record.get(key):
                    record[key] = json.dumps(record[key])
            records.append(record)
        return records

    def to_dataframe(self, **kwargs: Any) -> Any:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. pip install pandas")
        return pd.DataFrame(self._to_records(**kwargs))

    def to_csv(self, path: str, **kwargs: Any) -> None:
        import csv

        records = self._to_records(**kwargs)
        if not records:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def to_json(self, path: str, **kwargs: Any) -> None:
        exclude = {
            k for k, v in kwargs.items() if v is False and k != "mutated_description"
        }
        exclude_dict = {"cases": {"__all__": exclude}} if exclude else None
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, exclude=exclude_dict))

    def to_jsonl(self, path: str, **kwargs: Any) -> None:
        exclude = {
            k for k, v in kwargs.items() if v is False and k != "mutated_description"
        }
        with open(path, "w", encoding="utf-8") as f:
            for case in self.cases:
                f.write(
                    case.model_dump_json(exclude=exclude if exclude else None) + "\n"
                )

    def to_parquet(self, path: str, **kwargs: Any) -> None:
        df = self.to_dataframe(**kwargs)
        df.to_parquet(path)

    def sort_by(self, field: str, descending: bool = False) -> AugmentedDataset:
        if not self.cases:
            return self

        def _key(c: EvaluationCase) -> Any:
            val = getattr(c, field)
            if isinstance(val, Enum):
                return list(type(val)).index(val)
            return val

        sorted_cases = sorted(self.cases, key=_key, reverse=descending)
        return self.model_copy(update={"cases": sorted_cases})

    def filter(self, **kwargs: Any) -> AugmentedDataset:
        filtered = self.cases
        for key, value in kwargs.items():
            if key == "dimension":
                filtered = [
                    c
                    for c in filtered
                    if value.lower() in c.dimension_id.lower()
                    or value.lower() in c.dimension_name.lower()
                ]
            elif key == "severity":
                filtered = [
                    c for c in filtered if c.severity.value.lower() == value.lower()
                ]
            elif key == "keyword":
                keyword = value.lower()
                filtered = [
                    c
                    for c in filtered
                    if keyword in c.mutated_description.lower()
                    or keyword in c.rationale.lower()
                ]
            else:
                filtered = [c for c in filtered if getattr(c, key, None) == value]

        return self.model_copy(update={"cases": filtered})

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Any:
        return iter(self.cases)

    def __getitem__(self, index: int | slice) -> Any:
        return self.cases[index]
