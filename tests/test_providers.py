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
    assert result.intent != ""
    assert len(result.entities) > 0


def test_parse_json_strips_markdown_fences() -> None:
    content = '```json\n{"intent": "test", "entities": [], "constraints": [], "implicit_assumptions": [], "risk_areas": [], "edge_cases": [], "ambiguities": [], "domain_context": ""}\n```'
    result = MockLLMProvider._parse_json(content, BehaviorAnalysis)
    assert result.intent == "test"


def test_parse_json_raises_parse_error_on_bad_json() -> None:
    with pytest.raises(ParseError):
        MockLLMProvider._parse_json("not json at all !!!", BehaviorAnalysis)


def test_parse_json_extracts_json_from_free_text() -> None:
    content = 'Here is the analysis: {"intent": "x", "entities": [], "constraints": [], "implicit_assumptions": [], "risk_areas": [], "edge_cases": [], "ambiguities": [], "domain_context": ""} That is all.'
    result = MockLLMProvider._parse_json(content, BehaviorAnalysis)
    assert result.intent == "x"


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
