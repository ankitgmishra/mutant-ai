"""mutant/providers package — LLM provider abstraction layer."""

from mutant.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ParseError,
    ProviderError,
)

__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "GeminiProvider",
    "LLMMessage",
    "LLMResponse",
    "LiteLLMProvider",
    "OllamaProvider",
    # Concrete providers (imported lazily — optional deps)
    "OpenAIProvider",
    "ParseError",
    "ProviderError",
]


def __getattr__(name: str) -> object:
    """Lazy-import concrete providers to avoid ImportError when extras are missing."""
    _map = {
        "OpenAIProvider": ("mutant.providers.openai", "OpenAIProvider"),
        "AnthropicProvider": ("mutant.providers.anthropic", "AnthropicProvider"),
        "GeminiProvider": ("mutant.providers.gemini", "GeminiProvider"),
        "OllamaProvider": ("mutant.providers.ollama", "OllamaProvider"),
        "LiteLLMProvider": ("mutant.providers.litellm", "LiteLLMProvider"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'mutant.providers' has no attribute {name!r}")
