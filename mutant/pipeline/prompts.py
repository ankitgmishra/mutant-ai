"""
mutant/pipeline/prompts.py
==========================
Prompt template loader and renderer.
Uses Jinja2 to render markdown prompt templates.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@cache
def _get_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(
    template_name: str, template_override: str | None = None, **kwargs: Any
) -> str:
    """Render a prompt template from the prompts/ directory.

    Parameters
    ----------
    template_name:
        Filename of the template (e.g. ``"behavior_analysis.md"``).
    template_override:
        Optional raw string to use instead of the file.
    **kwargs:
        Template variables.

    Returns
    -------
    str
        The rendered prompt text.
    """
    env = _get_env()
    if template_override is not None:
        template = env.from_string(template_override)
    else:
        template = env.get_template(template_name)
    return template.render(**kwargs)
