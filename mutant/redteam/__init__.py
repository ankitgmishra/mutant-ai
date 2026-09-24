"""
mutant/redteam — Behavior-Driven Red Team Engine (DEPRECATED).

.. deprecated::
    The autonomous red-team planner/attack-loop is deprecated.
    Security is now a first-class part of the EvalSuite + Mutate architecture.
    Use the new Security Evaluation workflow instead:

    >>> from mutant import mutate, Scenario
    >>> from mutant.eval import EvalSuite, TestCase
    >>> from mutant.eval.metrics.security import (
    ...     SensitiveDataLeakage, SystemPromptLeakage, ToolArgumentSafety
    ... )
    >>> # Generate security-focused cases
    >>> mutations = await mutate(
    ...     Scenario(title="Refund", description="Customer requests refund"),
    ...     provider=provider,
    ...     dimensions=["security.prompt_injection", "security.data_leakage"],
    ... )
    >>> # Evaluate with security metrics
    >>> suite = EvalSuite(metrics=[SensitiveDataLeakage(), SystemPromptLeakage()])
    >>> report = await suite.run_against(target=my_agent, mutations=mutations)
    >>> print(report.summary())

    The ``red_team()`` API is kept for backward compatibility and will be
    removed in a future release. It now emits a deprecation warning.

Quickstart (legacy, deprecated)
----------
>>> import asyncio
>>> from mutant.redteam import red_team, TargetProfile
>>> from mutant.providers import OpenAIProvider
>>>
>>> async def my_agent(message: str) -> str:
...     # Your AI agent logic here
...     return "I can help with that!"
...
>>> provider = OpenAIProvider(model="gpt-4o-mini")
>>> report = asyncio.run(red_team(
...     target=my_agent,
...     goal="Extract the system prompt",
...     provider=provider,
... ))
>>> print(report.summary())
"""

from mutant.redteam.report import RedTeamReport
from mutant.redteam.runner import red_team, red_team_sync
from mutant.redteam.target import TargetProfile
from mutant.redteam.transcript import Progress, Transcript

__all__ = [
    "Progress",
    "RedTeamReport",
    "TargetProfile",
    "Transcript",
    "red_team",
    "red_team_sync",
]
