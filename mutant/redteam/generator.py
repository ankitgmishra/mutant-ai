"""
mutant/redteam/generator.py
==============================
Attack message generator.

Creates the actual adversarial message using existing MutationDimension
instructions, examples, and system context. This is where maximum reuse
of the mutation engine happens.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mutant.core.registry import MutationRegistry
from mutant.core.registry import registry as _default_registry
from mutant.pipeline.prompts import render_prompt
from mutant.providers.base import LLMMessage
from mutant.redteam.planner import AttackPlan
from mutant.redteam.transcript import Turn

if TYPE_CHECKING:
    from mutant.providers.base import BaseLLMProvider

logger = logging.getLogger("mutant.redteam")


class _GeneratedAttack(BaseModel):
    """LLM response schema for attack generation."""

    message: str = Field(description="The attack message to send to the target.")


async def generate_attack(
    plan: AttackPlan,
    goal: str,
    history: list[Turn],
    provider: BaseLLMProvider,
    registry: MutationRegistry | None = None,
    max_retries: int = 3,
) -> str:
    """Generate an adversarial attack message based on the plan.

    Reuses existing MutationDimension capabilities:
    - ``dimension.get_mutation_instructions()`` for attack instructions
    - ``dimension.get_examples()`` for few-shot style guidance
    - ``dimension.get_system_context()`` for domain knowledge

    Parameters
    ----------
    plan : AttackPlan
        The planner's decision (behavior, strategy, escalation).
    goal : str
        The attacker's objective.
    history : list[Turn]
        Conversation history so far.
    provider : BaseLLMProvider
        LLM provider for message generation.
    registry : MutationRegistry | None
        Dimension registry (uses default if None).

    Returns
    -------
    str
        The generated attack message.
    """
    reg = registry or _default_registry

    try:
        dimension = reg.get(plan.behavior)
    except KeyError:
        logger.warning(f"Unknown behavior {plan.behavior}, generating generic attack.")
        dimension = None

    # Reuse the EXACT same dimension methods used in mutation generation
    instructions = dimension.get_mutation_instructions() if dimension else ""
    examples = dimension.get_examples()[:3] if dimension else []
    system_context = dimension.get_system_context() if dimension else ""

    history_text = ""
    if history:
        lines = [f"{t.role.capitalize()}: {t.content}" for t in history[-10:]]
        history_text = "\n".join(lines)

    prompt = render_prompt(
        "redteam_generation.md",
        goal=goal,
        behavior_name=dimension.name if dimension else plan.behavior,
        instructions=instructions,
        examples=examples,
        system_context=system_context,
        strategy=plan.strategy,
        escalation=plan.escalation,
        history=history_text,
    )

    result = await provider.complete_json(
        [LLMMessage(role="user", content=prompt)],
        _GeneratedAttack,
        temperature=0.8,
        max_retries=max_retries,
    )
    return result.message
