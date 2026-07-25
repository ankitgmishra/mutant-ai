"""Scenario — the original behavioral situation under test."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Scenario(BaseModel):
    """Represents a single, original behavioral scenario to be mutated.

    A scenario is the atomic unit of input to Mutant. Developers write one
    realistic scenario; Mutant generates hundreds of behavioral mutations from it.

    Attributes
    ----------
    title:
        Short human-readable label for the scenario.
    description:
        Full description of the scenario. This is the text that mutations
        will be applied to.
    context:
        Optional extra metadata (agent name, domain, tags, etc.).
    tags:
        Free-form labels for filtering / grouping during reporting.

    Examples
    --------
    >>> s = Scenario(
    ...     title="Refund Request",
    ...     description="Customer bought a laptop. Requests a refund after 10 days.",
    ...     tags=["customer-support", "refund"],
    ... )
    >>> print(s.title)
    'Refund Request'
    """

    title: str = Field(..., min_length=1, description="Short human-readable label.")
    description: str = Field(
        ..., min_length=5, description="Full description of the scenario."
    )
    domain: str | None = Field(
        None, description="Optional explicit domain (e.g., healthcare, finance)."
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context (organization, jurisdiction, risk_level, compliance, agent_type, tools, etc.).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering / grouping.",
    )

    model_config = {"frozen": False, "extra": "forbid"}

    @model_validator(mode="after")
    def _normalise_tags(self) -> Scenario:
        self.tags = [t.lower().strip() for t in self.tags if t.strip()]
        return self

    # ── Convenience ───────────────────────────────────────────────────────────

    def with_description(self, description: str) -> Scenario:
        """Return a shallow copy with a new description (used internally by mutations)."""
        return self.model_copy(update={"description": description})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        """Create a Scenario from a dictionary, safely inferring missing fields."""
        title = data.pop("title", data.get("id", "Imported Scenario"))
        description = data.pop("description", data.pop("text", None))
        if not description:
            raise ValueError("Dictionary must contain a 'description' or 'text' key.")
        domain = data.pop("domain", None)
        tags = data.pop("tags", [])
        return cls(
            title=title, description=description, domain=domain, tags=tags, context=data
        )

    @classmethod
    def from_messages(
        cls, messages: list[dict[str, str]], title: str = "Chat Scenario"
    ) -> Scenario:
        """Create a Scenario from a list of chat messages."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return cls(
            title=title,
            description="\n\n".join(lines),
            context={"format": "chat_messages"},
        )

    @classmethod
    def from_chat(cls, chat_text: str, title: str = "Chat Scenario") -> Scenario:
        """Create a Scenario directly from raw chat text."""
        return cls(title=title, description=chat_text, context={"format": "raw_chat"})

    @classmethod
    def from_dataframe_row(
        cls, row: Any, text_column: str = "text", title_column: str | None = None
    ) -> Scenario:
        """Create a Scenario from a Pandas or Polars DataFrame row."""
        # Standard dict conversion handles pandas Series natively if using .to_dict() beforehand,
        # but if this is a raw Series or namedtuple from iterrows/itertuples:
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        elif hasattr(row, "_asdict"):
            data = row._asdict()
        elif isinstance(row, dict):
            data = dict(row)
        else:
            raise TypeError(
                "Row must be convertible to a dictionary (e.g. Pandas Series)."
            )

        title = (
            data.pop(title_column)
            if title_column and title_column in data
            else "Scenario from row"
        )
        description = data.pop(text_column, None)
        if not description:
            raise ValueError(f"Row missing required text_column: '{text_column}'")

        return cls(title=str(title), description=str(description), context=data)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Scenario(title={self.title!r}, description={self.description[:40]!r}...)"
        )
