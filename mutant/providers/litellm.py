"""LiteLLM provider — unified proxy for 100+ models."""

from __future__ import annotations

from typing import Any

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ProviderError,
)


class LiteLLMProvider(BaseLLMProvider):
    """LLM provider that wraps LiteLLM for access to 100+ models.

    Requires the ``litellm`` extra: ``pip install mutant-ai[litellm]``

    LiteLLM supports: OpenAI, Anthropic, Gemini, Azure, Cohere, Mistral,
    Together, Replicate, and many more — using a unified OpenAI-compatible API.

    Parameters
    ----------
    model:
        LiteLLM model string e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``,
        ``"gemini/gemini-1.5-pro"``, ``"ollama/llama3"``.
    api_key:
        API key (if not set via environment).
    kwargs:
        Any additional kwargs passed to ``litellm.acompletion``.

    Example
    -------
    >>> provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
    >>> cases = await mutate(scenario, provider=provider, count=50)
    """

    provider_name = "litellm"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LiteLLM provider requires the 'litellm' package. "
                "Install it with: pip install mutant-ai[litellm]"
            ) from exc

        self.model = model
        self._api_key = api_key
        self._extra_kwargs = kwargs

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **self._extra_kwargs,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage
            return LLMResponse(
                content=content,
                model=response.model or self.model,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        except Exception as exc:
            raise ProviderError(
                f"LiteLLM request failed: {exc}",
                provider=self.provider_name,
            ) from exc
