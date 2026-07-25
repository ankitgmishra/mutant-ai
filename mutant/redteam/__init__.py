"""
mutant/redteam — Behavior-Driven Red Team Engine.

Automatically discovers behavioral weaknesses in AI systems
through multi-turn adversarial conversations.

Quickstart
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
