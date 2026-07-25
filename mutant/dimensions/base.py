"""
mutant/dimensions/base.py
=========================
MutationDimension — the abstract base class for all mutation dimensions.

A dimension defines *how* to instruct the LLM to mutate a scenario.
It produces prompt fragments; the LLM produces the actual text.

This replaces the old rule-based ``Mutation.apply()`` pattern entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mutant.core.mutation import MutationCategory, MutationSeverity


class MutationDimension(ABC):
    """Abstract base class for all mutation dimensions.

    A dimension encodes domain knowledge about a class of behavioral
    perturbation. It does NOT generate text directly — it produces
    structured instructions that the LLM uses to generate mutations.

    To create a custom dimension::

        class LegalThreatDimension(MutationDimension):
            id = "custom.legal_threat"
            name = "Legal Threat"
            description = "Customer threatens legal action."
            category = MutationCategory.EMOTION
            severity = MutationSeverity.CRITICAL

            def get_mutation_instructions(self) -> str:
                return (
                    "Rewrite the scenario so the user explicitly threatens legal action, "
                    "such as contacting a lawyer, filing a lawsuit, or reporting to "
                    "consumer protection agencies. The threat must feel genuine and urgent."
                )

        registry.register(LegalThreatDimension())
    """

    # ── Required class-level attributes ───────────────────────────────────────

    id: str
    name: str
    description: str
    category: MutationCategory
    severity: MutationSeverity

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip validation for abstract intermediary classes
        if ABC in cls.__bases__:
            return
        required = ("id", "name", "description", "category", "severity")
        for attr in required:
            if not hasattr(cls, attr):
                raise TypeError(
                    f"{cls.__name__} must define class attribute {attr!r}. "
                    "See MutationDimension documentation for details."
                )

    @abstractmethod
    def get_mutation_instructions(self) -> str:
        """Return detailed LLM instructions for this mutation dimension.

        This text is injected directly into the mutation generation prompt.
        Write it as a precise, actionable directive to the LLM.

        Returns
        -------
        str
            A clear description of how the LLM should transform the scenario
            when this dimension is active.
        """

    def get_examples(self) -> list[tuple[str, str]]:
        """Return (original, mutated) example pairs for few-shot prompting.

        Optional. Override to provide concrete examples that anchor the LLM's
        generation style for this dimension.

        Returns
        -------
        list[tuple[str, str]]
            List of (original_description, mutated_description) pairs.
            2–3 pairs is ideal; more than 5 adds noise.
        """
        return []

    def get_system_context(self) -> str:
        """Return optional system-level context for this dimension.

        Override to provide domain knowledge the LLM should have when
        generating mutations for this dimension. E.g. legal terminology
        for a legal threats dimension.
        """
        return ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(id={self.id!r})"
