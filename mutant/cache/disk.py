"""Disk-based LRU cache using diskcache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from mutant.cache.base import BaseCache, CacheEntry, CacheKey


class DiskCache(BaseCache):
    """Persistent disk cache for LLM responses.

    Stores responses in a SQLite-backed diskcache database. This is the
    recommended cache for development workflows — it eliminates redundant
    LLM calls across runs.

    Parameters
    ----------
    directory:
        Path to the cache directory. Created if it does not exist.
        Default: ``~/.mutant/cache``
    size_limit:
        Maximum cache size in bytes. Default: 1 GB.
    ttl:
        Time-to-live in seconds. ``None`` means forever. Default: ``None``.

    Example
    -------
    >>> cache = DiskCache(directory=".mutant_cache")
    >>> cases = await mutate(scenario, provider=provider, cache=cache, count=50)
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        size_limit: int = 1024**3,
        ttl: float | None = None,
    ) -> None:
        try:
            import diskcache
        except ImportError as exc:
            raise ImportError(
                "DiskCache requires the 'diskcache' package (included in mutant-ai core)."
            ) from exc

        import diskcache

        path = Path(directory) if directory else Path.home() / ".mutant" / "cache"
        path.mkdir(parents=True, exist_ok=True)
        self._cache: Any = diskcache.Cache(str(path), size_limit=size_limit)
        self._ttl = ttl

    async def get(self, key: CacheKey) -> CacheEntry | None:
        k = key.as_string()
        raw = await anyio.to_thread.run_sync(lambda: self._cache.get(k))
        if raw is None:
            return None
        return CacheEntry.model_validate(raw)

    async def set(self, entry: CacheEntry) -> None:
        k = entry.key.as_string()
        data = entry.model_dump()
        await anyio.to_thread.run_sync(
            lambda: self._cache.set(k, data, expire=self._ttl)
        )

    async def clear(self) -> None:
        await anyio.to_thread.run_sync(self._cache.clear)

    def __len__(self) -> int:
        return len(self._cache)  # type: ignore[arg-type]
