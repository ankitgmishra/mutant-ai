"""Gemini provider implementation."""

from __future__ import annotations

from typing import Any

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ProviderError,
)


class GeminiProvider(BaseLLMProvider):
    """LLM provider for Google Gemini models.

    Requires the ``gemini`` extra: ``pip install mutant-ai[gemini]``

    Parameters
    ----------
    api_key:
        Google API key. Defaults to ``GOOGLE_API_KEY`` env var.
    model:
        Model identifier. Default: ``"gemini-1.5-flash"``.
    """

    provider_name = "gemini"

    # Global state for free-tier rate limiting
    _global_lock = None
    _last_request_time = 0.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires the 'google-genai' package. "
                "Install it with: pip install google-genai"
            ) from exc

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self._genai = genai

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> LLMResponse:

        # Build prompt from messages
        parts: list[str] = []
        for m in messages:
            prefix = "" if m.role == "user" else f"[{m.role.upper()}] "
            parts.append(prefix + m.content)
        prompt = "\n\n".join(parts)

        import asyncio
        import time

        from tenacity import (
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_random_exponential,
        )

        # Initialize the global lock once per event loop
        if GeminiProvider._global_lock is None:
            GeminiProvider._global_lock = asyncio.Lock()

        def is_retryable(exc: BaseException) -> bool:
            exc_str = str(exc).lower()
            return (
                "429" in exc_str
                or "quota" in exc_str
                or "resourceexhausted" in exc_str
                or "too many requests" in exc_str
            )

        def before_sleep_print(retry_state: Any) -> None:
            exc = retry_state.outcome.exception()
            print(
                f"[WARNING] Gemini API Error ({type(exc).__name__}: {exc}). Sleeping {retry_state.next_action.sleep:.1f}s before retry "
                f"(Attempt {retry_state.attempt_number}/10)..."
            )

        @retry(
            wait=wait_random_exponential(multiplier=2, max=65),
            stop=stop_after_attempt(10),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_print,
            reraise=True,
        )
        async def _generate() -> LLMResponse:
            try:
                # Enforce a strict global rate limit for free tier (approx 1 request per 4 seconds)
                async with GeminiProvider._global_lock:
                    now = time.monotonic()
                    time_since_last = now - GeminiProvider._last_request_time
                    if time_since_last < 4.5:
                        await asyncio.sleep(4.5 - time_since_last)

                    GeminiProvider._last_request_time = time.monotonic()

                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._genai.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return LLMResponse(
                    content=response.text,
                    model=self.model_name,
                )
            except Exception as e:
                if is_retryable(e):
                    raise
                raise ProviderError(
                    f"Gemini request failed: {e}",
                    provider=self.provider_name,
                ) from e

        return await _generate()
