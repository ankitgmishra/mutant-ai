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
        """Structured completion — Ollama is constrained to a JSON object.

        The mutation engine and the metrics' LLM judges parse this output, so the
        constraint is deliberate. For prose use :meth:`complete_text`.
        """
        return await self._chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

    async def complete_text(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Natural-language completion — no ``format`` constraint.

        Use this for anything whose output is read by a human or by an
        application: agent replies, RAG answers, refusals. Constraining these to
        JSON makes the answer arrive as ``{"role": ..., "content": ...}`` and can
        turn a refusal into the user's own message echoed back.
        """
        return await self._chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )

    async def _chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            # qwen3:4b is a thinking model — disable thinking for structured JSON tasks
            # Ollama respects "think": false (new) and ignores it for non-thinking models
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {}) or {}
                content = msg.get("content", "") or ""
                # Fallback: Ollama thinking models may return thinking separately
                if not content:
                    thinking = msg.get("thinking", "") or data.get("thinking", "") or ""
                    if thinking:
                        # If thinking contains JSON, use it; else mark as empty for retry
                        content = thinking
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
