"""Tests for the provider abstraction."""

from __future__ import annotations

import pytest

from mutant.core.mutation import BehaviorAnalysis
from mutant.providers.base import BaseLLMProvider, LLMMessage, LLMResponse, ParseError
from tests.conftest import MockLLMProvider


def test_mock_provider_is_valid_provider() -> None:
    provider = MockLLMProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.provider_name == "mock"


@pytest.mark.asyncio
async def test_mock_provider_complete_returns_response() -> None:
    provider = MockLLMProvider()
    response = await provider.complete([LLMMessage(role="user", content="Hello")])
    assert isinstance(response, LLMResponse)
    assert response.model == "mock-model"


@pytest.mark.asyncio
async def test_mock_provider_complete_json_returns_typed_model() -> None:
    provider = MockLLMProvider()
    result = await provider.complete_json(
        [LLMMessage(role="user", content="Analyze this")],
        BehaviorAnalysis,
    )
    assert isinstance(result, BehaviorAnalysis)
    assert result.detected_domain != ""
    assert len(result.entities) > 0


def test_parse_json_strips_markdown_fences() -> None:
    content = '```json\n{"detected_domain": "test"}\n```'
    result = MockLLMProvider._parse_json(content, BehaviorAnalysis)
    assert result.detected_domain == "test"


def test_parse_json_raises_parse_error_on_bad_json() -> None:
    with pytest.raises(ParseError):
        MockLLMProvider._parse_json("not json at all !!!", BehaviorAnalysis)


def test_parse_json_extracts_json_from_free_text() -> None:
    content = 'Here is the analysis: {"detected_domain": "x"} That is all.'
    result = MockLLMProvider._parse_json(content, BehaviorAnalysis)
    assert result.detected_domain == "x"


def test_mock_provider_call_count_increments() -> None:
    """Verify test isolation — call count tracks invocations."""
    import asyncio

    provider = MockLLMProvider()
    assert provider.call_count == 0
    asyncio.run(
        provider.complete_json(
            [LLMMessage(role="user", content="test")],
            BehaviorAnalysis,
        )
    )
    assert provider.call_count == 1


def test_llm_response_total_tokens() -> None:
    r = LLMResponse(content="hi", model="test", input_tokens=10, output_tokens=20)
    assert r.total_tokens == 30


def test_llm_response_total_tokens_none_when_missing() -> None:
    r = LLMResponse(content="hi", model="test")
    assert r.total_tokens is None


@pytest.mark.asyncio
async def test_base_complete_text_delegates_to_complete() -> None:
    """Providers that do not force a format need no override."""
    provider = MockLLMProvider()
    response = await provider.complete_text([LLMMessage(role="user", content="Hello")])
    assert isinstance(response, LLMResponse)
    assert provider.call_count == 1


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": "answer"}, "prompt_eval_count": 1, "eval_count": 2}


class _RecordingAsyncClient:
    """Captures the payload OllamaProvider posts, without touching the network."""

    last_payload: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, json=None):
        _RecordingAsyncClient.last_payload = json
        return _FakeResponse()


@pytest.mark.asyncio
async def test_ollama_complete_constrains_output_to_json(monkeypatch) -> None:
    """The mutation engine and the LLM judges parse this output, so it stays JSON."""
    from mutant.providers import ollama as ollama_module

    _RecordingAsyncClient.last_payload = None
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", _RecordingAsyncClient)

    provider = ollama_module.OllamaProvider(model="test-model")
    await provider.complete([LLMMessage(role="user", content="hi")])

    assert _RecordingAsyncClient.last_payload is not None
    assert _RecordingAsyncClient.last_payload["format"] == "json"


@pytest.mark.asyncio
async def test_ollama_complete_text_omits_json_format(monkeypatch) -> None:
    """Prose must not be constrained: an envelope would break agent replies and
    can turn a refusal into the user's own message echoed back."""
    from mutant.providers import ollama as ollama_module

    _RecordingAsyncClient.last_payload = None
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", _RecordingAsyncClient)

    provider = ollama_module.OllamaProvider(model="test-model")
    await provider.complete_text([LLMMessage(role="user", content="hi")])

    assert _RecordingAsyncClient.last_payload is not None
    assert "format" not in _RecordingAsyncClient.last_payload

