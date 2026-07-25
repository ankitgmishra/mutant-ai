"""MutationRegistry — pluggable dimension store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

from mutant.core.mutation import MutationCategory

if TYPE_CHECKING:
    from mutant.dimensions.base import MutationDimension


class MutationRegistry:
    """Thread-safe registry for :class:`~mutant.dimensions.base.MutationDimension` objects.

    The registry is the single source of truth for which mutation dimensions
    are available. Built-in dimensions are auto-registered when
    ``mutant.dimensions`` is imported. Third-party packages register their
    own dimensions here.

    Example
    -------
    >>> from mutant import registry
    >>> from mutant.dimensions import MutationDimension
    >>> from mutant.core.mutation import MutationCategory, MutationSeverity
    >>>
    >>> class LegalThreatDimension(MutationDimension):
    ...     id = "custom.legal_threat"
    ...     name = "Legal Threat"
    ...     description = "Customer threatens legal action."
    ...     category = MutationCategory.EMOTION
    ...     severity = MutationSeverity.CRITICAL
    ...     def get_mutation_instructions(self):
    ...         return "Rewrite so the customer explicitly threatens legal action."
    >>>
    >>> registry.register(LegalThreatDimension())
    >>> "custom.legal_threat" in registry
    True
    """

    def __init__(self) -> None:
        self._dimensions: dict[str, MutationDimension] = {}
        self._by_category: dict[MutationCategory, list[MutationDimension]] = (
            defaultdict(list)
        )

    # ── Dimension management ───────────────────────────────────────────────────

    def register(self, dimension: MutationDimension) -> None:
        """Register a mutation dimension.

        Parameters
        ----------
        dimension:
            An instantiated :class:`~mutant.dimensions.base.MutationDimension` subclass.

        Raises
        ------
        ValueError
            If a dimension with the same ``id`` is already registered.
        """
        if dimension.id in self._dimensions:
            raise ValueError(
                f"A dimension with id={dimension.id!r} is already registered. "
                "Use registry.unregister() first if you want to replace it."
            )
        self._dimensions[dimension.id] = dimension
        self._by_category[dimension.category].append(dimension)

    def unregister(self, dimension_id: str) -> None:
        """Remove a dimension by id.

        Raises
        ------
        KeyError
            If no dimension with that id exists.
        """
        dim = self._dimensions.pop(dimension_id)
        self._by_category[dim.category].remove(dim)

    def get(self, dimension_id: str) -> MutationDimension:
        """Retrieve a dimension by id.

        Raises
        ------
        KeyError
            If the id is not found.
        """
        try:
            return self._dimensions[dimension_id]
        except KeyError:
            raise KeyError(
                f"No dimension registered with id={dimension_id!r}."
            ) from None

    # ── Querying ───────────────────────────────────────────────────────────────

    def all(self) -> list[MutationDimension]:
        """Return all registered dimensions."""
        return list(self._dimensions.values())

    def by_category(self, category: MutationCategory) -> list[MutationDimension]:
        """Return all dimensions in a given category."""
        return list(self._by_category[category])

    def categories(self) -> list[MutationCategory]:
        """Return all categories that have at least one dimension registered."""
        return [cat for cat, dims in self._by_category.items() if dims]

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __contains__(self, dimension_id: str) -> bool:
        return dimension_id in self._dimensions

    def __len__(self) -> int:
        return len(self._dimensions)

    def __iter__(self) -> Iterator[MutationDimension]:
        return iter(self._dimensions.values())

    def __repr__(self) -> str:  # pragma: no cover
        return f"MutationRegistry(count={len(self)})"


# ── Singleton default registry ─────────────────────────────────────────────────
registry = MutationRegistry()
