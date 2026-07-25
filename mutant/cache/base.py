"""Cache base classes."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class CacheKey(BaseModel):
    """A deterministic cache key built from prompt content and parameters."""

    provider: str
    model: str
    messages_hash: str  # SHA256 of serialised messages
    temperature: float

    model_config = {"frozen": True}

    @classmethod
    def from_request(
        cls,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> CacheKey:
        payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return cls(
            provider=provider,
            model=model,
            messages_hash=digest,
            temperature=temperature,
        )

    def as_string(self) -> str:
        return f"{self.provider}:{self.model}:{self.messages_hash}:{self.temperature}"


class CacheEntry(BaseModel):
    """A cached LLM response."""

    key: CacheKey
    content: str
    metadata: dict[str, Any] = {}


class BaseCache(ABC):
    """Abstract base class for response caches.

    Implement ``get`` and ``set`` to create a custom cache backend.
    """

    @abstractmethod
    async def get(self, key: CacheKey) -> CacheEntry | None:
        """Return a cached entry, or ``None`` if not found."""

    @abstractmethod
    async def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
