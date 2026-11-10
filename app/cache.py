"""Query cache: skip retrieval and the Claude call for a question already answered.

Why cache at all
----------------
Generation is the slow, paid part of a query: a second or more and a model
call per request, against a few milliseconds for retrieval. Document Q&A sees
the same questions repeatedly ("how do I reset my password?"), so answering
each one once and serving it from Redis afterwards cuts both cost and latency.

Why exact match, not a semantic cache
-------------------------------------
A semantic cache serves a stored answer to any question whose embedding is
close enough. That catches rewordings, but "how do I enable 2FA?" and "how do I
disable 2FA?" embed very close together and need opposite answers. Serving a
confidently wrong answer is worse than paying for a fresh one, so this cache
only hits on the same question (after normalising case and whitespace) with
the same retrieval settings. A semantic layer could sit on top later, gated by
a measured similarity threshold.

Staying correct when the index changes
--------------------------------------
Every key includes a generation number. Ingesting or deleting a document bumps
the generation, so every earlier entry stops matching at once, and the old
entries expire on their own via TTL. That is one O(1) write instead of
scanning Redis for keys to delete.

A cache must never take the service down: any Redis error is logged and
treated as a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional, Protocol

log = logging.getLogger("engine.cache")

_WHITESPACE = re.compile(r"\s+")


def normalise_question(question: str) -> str:
    """Case and spacing don't change the answer; everything else might."""
    return _WHITESPACE.sub(" ", question).strip().lower()


def cache_key(generation: int, **params: Any) -> str:
    """A stable key for one question under one set of retrieval settings."""
    params["question"] = normalise_question(params.get("question", ""))
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:32]
    return f"engine:q:{generation}:{digest}"


class QueryCache(Protocol):
    def generation(self) -> int: ...
    def get(self, key: str) -> Optional[dict]: ...
    def set(self, key: str, value: dict) -> None: ...
    def invalidate(self) -> None: ...


class NullCache:
    """Caching disabled: every lookup misses, nothing is stored."""

    def generation(self) -> int:
        return 0

    def get(self, key: str) -> Optional[dict]:
        return None

    def set(self, key: str, value: dict) -> None:
        pass

    def invalidate(self) -> None:
        pass


class MemoryCache:
    """Single-process cache with the same behaviour as RedisCache, for tests."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._gen = 0
        self._items: dict[str, tuple[float, dict]] = {}

    def generation(self) -> int:
        return self._gen

    def get(self, key: str) -> Optional[dict]:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if time.monotonic() > expires:
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict) -> None:
        self._items[key] = (time.monotonic() + self.ttl, value)

    def invalidate(self) -> None:
        self._gen += 1


class RedisCache:  # pragma: no cover - exercised by tests/test_cache.py against a live Redis
    """Shared cache in Redis, so every API replica serves the same entries."""

    GENERATION_KEY = "engine:q:generation"

    def __init__(self, url: str, ttl_seconds: int = 3600) -> None:
        import redis

        self.ttl = ttl_seconds
        # Short timeouts: a slow cache should cost a few milliseconds, not hold
        # a query hostage.
        self._redis = redis.Redis.from_url(
            url, socket_timeout=0.5, socket_connect_timeout=0.5
        )

    def generation(self) -> int:
        try:
            value = self._redis.get(self.GENERATION_KEY)
            return int(value) if value else 0
        except Exception as exc:
            log.warning("cache unavailable, serving uncached: %s", exc)
            return -1   # a generation no entry was ever written under

    def get(self, key: str) -> Optional[dict]:
        try:
            raw = self._redis.get(key)
        except Exception as exc:
            log.warning("cache read failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict) -> None:
        try:
            self._redis.set(key, json.dumps(value), ex=self.ttl)
        except Exception as exc:
            log.warning("cache write failed: %s", exc)

    def invalidate(self) -> None:
        try:
            self._redis.incr(self.GENERATION_KEY)
        except Exception as exc:
            log.warning("cache invalidation failed: %s", exc)


def build_cache(redis_url: str, ttl_seconds: int) -> QueryCache:
    """Redis when REDIS_URL is set, otherwise no caching."""
    if not redis_url:
        return NullCache()
    return RedisCache(redis_url, ttl_seconds)  # pragma: no cover - needs Redis
