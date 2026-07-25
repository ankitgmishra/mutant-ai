"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ParseError,
    ProviderError,
)

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI (GPT-4o, GPT-4-turbo, o1, etc.).

    Requires the ``openai`` extra: ``pip install mutant-ai[openai]``

    Parameters
    ----------
    api_key:
        OpenAI API key. Defaults to ``OPENAI_API_KEY`` environment variable.
    model:
        Model identifier. Default: ``"gpt-4o-mini"``.
    base_url:
        Override for OpenAI-compatible endpoints (e.g. Azure, local proxies).
    default_headers:
        Extra headers to send with every request.

    Example
    -------
    >>> provider = OpenAIProvider(model="gpt-4o")
    >>> cases = await mutate(scenario, provider=provider, count=50)
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install mutant-ai[openai]"
            ) from exc

        import openai as _openai

        self.model = model
        self._client = _openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or {},
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = response.usage
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            )
        except Exception as exc:
            raise ProviderError(
                f"OpenAI request failed: {exc}",
                provider=self.provider_name,
            ) from exc

    async def complete_json(
        self,
        messages: list[LLMMessage],
        schema: type[T],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> T:
        """Uses OpenAI JSON mode for reliable structured output."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_json(content, schema)
        except ParseError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"OpenAI JSON request failed: {exc}",
                provider=self.provider_name,
            ) from exc
