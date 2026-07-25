"""
mutant/coverage/taxonomy.py
===========================
Coverage taxonomy that directly mirrors the MutationRegistry.
This ensures mutate() and coverage() speak the exact same language.
"""

from __future__ import annotations

from mutant.core.registry import MutationRegistry
from mutant.core.registry import registry as default_mutation_registry


class BehaviorTaxonomy:
    """A behavioral taxonomy backed directly by a MutationRegistry.

    Parameters
    ----------
    registry:
        The MutationRegistry to use. Defaults to the global registry.
    """

    def __init__(self, registry: MutationRegistry | None = None) -> None:
        self.registry = registry or default_mutation_registry

    @property
    def total_behaviors(self) -> int:
        return len(self.registry.all())

    @property
    def all_dimension_ids(self) -> list[str]:
        return [d.id for d in self.registry.all()]


DEFAULT_TAXONOMY = BehaviorTaxonomy()
