"""mutant/cache — pluggable response caching for LLM calls."""

from mutant.cache.base import BaseCache, CacheEntry, CacheKey
from mutant.cache.disk import DiskCache

__all__ = ["BaseCache", "CacheEntry", "CacheKey", "DiskCache"]
