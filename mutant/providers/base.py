"""
mutant/providers/base.py
========================
Provider-agnostic LLM abstraction.

No provider-specific code leaks beyond this boundary. The rest of Mutant
only ever sees ``BaseLLMProvider``, ``LLMMessage``, and ``LLMResponse``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    """The response from an LLM provider."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = {}

    model_config = {"frozen": True}

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class ProviderError(Exception):
    """Raised when an LLM provider returns an error."""

    def __init__(
        self, message: str, *, provider: str, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ParseError(Exception):
    """Raised when structured output parsing fails."""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Implementors must only override ``complete()``. JSON extraction and
    structured parsing are handled by this base class.

    Example
    -------
    >>> provider = OpenAIProvider(api_key="sk-...")
    >>> response = await provider.complete([
    ...     LLMMessage(role="user", content="Hello!")
    ... ])
    >>> print(response.content)
    """

    # Subclasses should set this.
    provider_name: str = "unknown"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send messages to the provider and return its response.

        Parameters
        ----------
        messages:
            Conversation history. Use ``role="system"`` for system prompts.
        temperature:
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens:
            Maximum tokens in the response.
        """

    async def complete_json(
        self,
        messages: list[LLMMessage],
        schema: type[T],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> T:
        """Complete and parse the response as a Pydantic model.

        The default implementation appends a JSON reminder to the last user
        message, completes, then extracts and validates JSON. Providers that
        natively support structured output (e.g. OpenAI ``response_format``)
        can override this for reliability.

        Parameters
        ----------
        messages:
            Conversation messages.
        schema:
            The Pydantic ``BaseModel`` subclass to parse into.

        Raises
        ------
        ParseError
            If the response cannot be parsed as valid JSON or validated
            against ``schema``.
        """
        last_error = None
        for _attempt in range(max_retries):
            response = await self.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return self._parse_json(response.content, schema)
            except ParseError as e:
                last_error = e
                # Adjust temperature slightly for retries to get a different result
                temperature = min(1.0, temperature + 0.1)

        raise last_error  # type: ignore

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(content: str, schema: type[T]) -> T:
        """Extract JSON from content and validate against schema."""
        # Strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first and last fence lines
            inner = (
                "\n".join(lines[1:-1])
                if lines[-1].strip() == "```"
                else "\n".join(lines[1:])
            )
            text = inner.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to find JSON block within free text
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise ParseError(
                        f"Could not parse JSON from LLM response: {exc}",
                        raw_content=content,
                    ) from exc
            else:
                raise ParseError(
                    f"No JSON found in LLM response: {exc}",
                    raw_content=content,
                ) from exc

        try:
            return schema.model_validate(data)
        except Exception as exc:
            raise ParseError(
                f"LLM response did not match schema {schema.__name__}: {exc}",
                raw_content=content,
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(provider={self.provider_name!r})"
