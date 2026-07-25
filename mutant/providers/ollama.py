"""Ollama provider — local LLM inference via Ollama's REST API."""

from __future__ import annotations

from typing import Any

import httpx

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ProviderError,
)


class OllamaProvider(BaseLLMProvider):
    """LLM provider for locally-running Ollama models.

    No extra packages required — uses ``httpx`` (a core dependency).

    Parameters
    ----------
    model:
        Ollama model name. Default: ``"llama3.1"``.
    base_url:
        Ollama server URL. Default: ``"http://localhost:11434"``.
    timeout:
        HTTP timeout in seconds. Default: ``120``.

    Example
    -------
    >>> provider = OllamaProvider(model="llama3.1")
    >>> cases = await mutate(scenario, provider=provider, count=20)
    """

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return LLMResponse(
                    content=content,
                    model=self.model,
                    input_tokens=data.get("prompt_eval_count"),
                    output_tokens=data.get("eval_count"),
                )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama request failed: {exc}",
                provider=self.provider_name,
            ) from exc
