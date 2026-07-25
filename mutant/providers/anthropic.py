"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ProviderError,
)

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider for Anthropic Claude models.

    Requires the ``anthropic`` extra: ``pip install mutant-ai[anthropic]``

    Parameters
    ----------
    api_key:
        Anthropic API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
    model:
        Model identifier. Default: ``"claude-3-5-haiku-20241022"``.

    Example
    -------
    >>> provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
    >>> cases = await mutate(scenario, provider=provider, count=50)
    """

    provider_name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install mutant-ai[anthropic]"
            ) from exc

        import anthropic as _anthropic

        self.model = model
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Separate system message from the rest
        system_content = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                user_messages.append({"role": m.role, "content": m.content})

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages,
            }
            if system_content:
                kwargs["system"] = system_content

            response = await self._client.messages.create(**kwargs)
            content = response.content[0].text if response.content else ""
            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except Exception as exc:
            raise ProviderError(
                f"Anthropic request failed: {exc}",
                provider=self.provider_name,
            ) from exc
