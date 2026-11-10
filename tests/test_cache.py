"""Query cache: keys, invalidation, the API wiring, and Redis when available."""

import io
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app import cache as cache_module
from app import main
from app.cache import MemoryCache, NullCache, cache_key, normalise_question
from app.config import settings


# ------------------------------------------------------------------- keys


def test_normalise_ignores_case_and_spacing():
    assert normalise_question("  What is   ERR_4032? ") == "what is err_4032?"


def test_same_request_same_key():
    a = cache_key(3, question="What is ERR_4032?", top_k=8, method="hybrid")
    b = cache_key(3, question="what is  err_4032?", top_k=8, method="hybrid")
    assert a == b


@pytest.mark.parametrize("change", [
    {"question": "What is ERR_4033?"},
    {"top_k": 5},
    {"method": "dense"},
])
def test_anything_that_changes_the_answer_changes_the_key(change):
    base = {"question": "What is ERR_4032?", "top_k": 8, "method": "hybrid"}
    assert cache_key(3, **base) != cache_key(3, **{**base, **change})


def test_generation_is_part_of_the_key():
    assert cache_key(1, question="q") != cache_key(2, question="q")


# ------------------------------------------------------------- backends


def test_null_cache_never_hits():
    cache = NullCache()
    cache.set("k", {"a": 1})
    assert cache.get("k") is None


def test_memory_cache_round_trip_and_invalidate():
    cache = MemoryCache()
    key = cache_key(cache.generation(), question="q")
    cache.set(key, {"answer": "x"})
    assert cache.get(key) == {"answer": "x"}

    cache.invalidate()
    assert cache.get(cache_key(cache.generation(), question="q")) is None


def test_memory_cache_expires(monkeypatch):
    cache = MemoryCache(ttl_seconds=10)
    cache.set("k", {"a": 1})
    real = time.monotonic()
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: real + 11)
    assert cache.get("k") is None


# ------------------------------------------------------------------- API

DOC = """# Error Reference

## ERR_4032

ERR_4032 is emitted when the connection pool is exhausted and no handle becomes
free within the wait threshold.
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(main, "build_cache", lambda url, ttl: MemoryCache(ttl))
    with TestClient(main.app) as c:
        yield c


def ingest(client, name="a.md", text=DOC):
    job = client.post(
        "/documents",
        files={"file": (name, io.BytesIO(text.encode()), "text/markdown")},
    ).json()
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/documents/{job['job_id']}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("ingestion did not finish")


def ask(client, question="What is ERR_4032?"):
    return client.post("/query", json={"question": question, "generate": False}).json()


def test_repeated_question_is_served_from_cache(client):
    ingest(client)
    first, second = ask(client), ask(client)
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["retrieved"] == first["retrieved"]


def test_rewording_case_and_spacing_still_hits(client):
    ingest(client)
    ask(client, "What is ERR_4032?")
    assert ask(client, "  what is err_4032? ")["cached"] is True


def test_ingesting_a_document_invalidates_the_cache(client):
    ingest(client)
    ask(client)
    ingest(client, "b.md", DOC.replace("ERR_4032", "ERR_9999"))
    assert ask(client)["cached"] is False


def test_deleting_a_document_invalidates_the_cache(client):
    first = ingest(client)
    ingest(client, "b.md")
    ask(client)
    assert client.delete(f"/documents/{first['document_id']}").status_code == 200
    assert ask(client)["cached"] is False


# ------------------------------------------------------------ live Redis

REDIS_URL = os.environ.get("TEST_REDIS_URL")


@pytest.mark.skipif(not REDIS_URL, reason="set TEST_REDIS_URL to run against Redis")
def test_redis_cache_round_trip_and_invalidate():
    from app.cache import RedisCache

    cache = RedisCache(REDIS_URL, ttl_seconds=30)
    question = f"q-{uuid.uuid4()}"
    key = cache_key(cache.generation(), question=question)
    cache.set(key, {"answer": "x"})
    assert cache.get(key) == {"answer": "x"}

    cache.invalidate()
    assert cache.get(cache_key(cache.generation(), question=question)) is None


@pytest.mark.skipif(not REDIS_URL, reason="set TEST_REDIS_URL to run against Redis")
def test_redis_outage_degrades_to_a_miss():
    from app.cache import RedisCache

    cache = RedisCache("redis://127.0.0.1:1/0")    # nothing listens on port 1
    assert cache.generation() == -1
    assert cache.get("anything") is None
    cache.set("anything", {"a": 1})               # must not raise
    cache.invalidate()                            # must not raise
