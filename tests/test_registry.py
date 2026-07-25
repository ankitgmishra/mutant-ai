"""Tests for MutationRegistry with MutationDimension."""

from __future__ import annotations

import pytest

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.core.registry import MutationRegistry
from mutant.dimensions.base import MutationDimension


class _AlphaDimension(MutationDimension):
    id = "test.alpha"
    name = "Alpha"
    description = "Alpha test dimension."
    category = MutationCategory.CONTEXT
    severity = MutationSeverity.LOW

    def get_mutation_instructions(self) -> str:
        return "Apply alpha mutation."


class _BetaDimension(MutationDimension):
    id = "test.beta"
    name = "Beta"
    description = "Beta test dimension."
    category = MutationCategory.EMOTION
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return "Apply beta mutation."


def test_register_and_contains(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    assert "test.alpha" in fresh_registry


def test_register_duplicate_raises(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register(_AlphaDimension())


def test_unregister(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    fresh_registry.unregister("test.alpha")
    assert "test.alpha" not in fresh_registry


def test_unregister_missing_raises(fresh_registry: MutationRegistry) -> None:
    with pytest.raises(KeyError):
        fresh_registry.unregister("does.not.exist")


def test_get(fresh_registry: MutationRegistry) -> None:
    dim = _AlphaDimension()
    fresh_registry.register(dim)
    assert fresh_registry.get("test.alpha") is dim


def test_get_missing_raises(fresh_registry: MutationRegistry) -> None:
    with pytest.raises(KeyError):
        fresh_registry.get("nope")


def test_all(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    fresh_registry.register(_BetaDimension())
    assert len(fresh_registry.all()) == 2


def test_by_category(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    fresh_registry.register(_BetaDimension())
    ctx_dims = fresh_registry.by_category(MutationCategory.CONTEXT)
    assert len(ctx_dims) == 1
    assert ctx_dims[0].id == "test.alpha"


def test_len(fresh_registry: MutationRegistry) -> None:
    assert len(fresh_registry) == 0
    fresh_registry.register(_AlphaDimension())
    assert len(fresh_registry) == 1


def test_iter(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    fresh_registry.register(_BetaDimension())
    ids = {d.id for d in fresh_registry}
    assert ids == {"test.alpha", "test.beta"}


def test_categories(fresh_registry: MutationRegistry) -> None:
    fresh_registry.register(_AlphaDimension())
    cats = fresh_registry.categories()
    assert MutationCategory.CONTEXT in cats
    assert MutationCategory.EMOTION not in cats


def test_global_registry_has_47_dimensions() -> None:
    import mutant  # noqa: F401 — triggers registration
    from mutant.core.registry import registry

    assert len(registry) == 47
