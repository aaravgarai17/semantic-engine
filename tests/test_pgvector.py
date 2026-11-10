"""PgVectorStore against a real Postgres with pgvector.

Skipped unless TEST_DATABASE_URL points at a database where CREATE EXTENSION
vector is allowed. CI provides one; locally:

    docker compose up -d db
    TEST_DATABASE_URL=postgresql://engine:engine@localhost:5432/engine pytest tests/test_pgvector.py
"""

import io
import os
import time

import pytest

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="set TEST_DATABASE_URL to run")

from app.embeddings import HashEmbedder  # noqa: E402
from app.store import StoredChunk  # noqa: E402

DIMS = 64


def reset(url):
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")


@pytest.fixture
def store():
    from app.store import PgVectorStore

    reset(DATABASE_URL)
    s = PgVectorStore(DATABASE_URL, DIMS)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return HashEmbedder(dimensions=DIMS)


def chunk(n, text, doc="doc1", collection="default", title="Runbook"):
    return StoredChunk(
        chunk_id=f"{doc}:{n}", document_id=doc, collection=collection, text=text,
        embed_text=text, index=n, heading=f"Section {n}", page=n + 1,
        document_title=title,
    )


TEXTS = [
    "ERR_4032 is raised when the connection pool is exhausted.",
    "Install libpq-dev before building the Postgres driver.",
    "Index vectors with vector_cosine_ops for cosine distance.",
    "Password resets send a single-use link to the email on file.",
]


def load(store, embedder, texts=TEXTS, **kw):
    chunks = [chunk(i, t, **kw) for i, t in enumerate(texts)]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))
    return chunks


def test_round_trip_keeps_every_field(store, embedder):
    load(store, embedder)
    got = store.get("doc1:1")
    assert got == chunk(1, TEXTS[1])
    assert got.citation() == "Runbook · p.2 · Section 1"


def test_get_many_preserves_requested_order(store, embedder):
    load(store, embedder)
    ids = ["doc1:2", "doc1:0", "missing", "doc1:3"]
    assert [c.chunk_id for c in store.get_many(ids)] == ["doc1:2", "doc1:0", "doc1:3"]


def test_dense_search_finds_the_exact_chunk_first(store, embedder):
    load(store, embedder)
    hits = store.search_dense(embedder.embed_query(TEXTS[2]), top_k=3)
    assert hits[0].chunk_id == "doc1:2"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert [h.rank for h in hits] == [1, 2, 3]


@pytest.mark.parametrize("query, expected", [
    ("what does ERR_4032 mean", "doc1:0"),
    ("libpq-dev", "doc1:1"),
    ("vector_cosine_ops index", "doc1:2"),
])
def test_keyword_search_matches_identifiers(store, embedder, query, expected):
    load(store, embedder)
    hits = store.search_keyword(query, top_k=3)
    assert hits and hits[0].chunk_id == expected


def test_keyword_search_matches_any_term_not_all(store, embedder):
    # A question with words no chunk contains must still find the chunk that
    # shares some of them; an AND query would return nothing.
    load(store, embedder)
    hits = store.search_keyword("how do I reset a forgotten password quickly", top_k=3)
    assert hits and hits[0].chunk_id == "doc1:3"


def test_keyword_search_with_only_stopwords_returns_nothing(store, embedder):
    load(store, embedder)
    assert store.search_keyword("the and of", top_k=3) == []


def test_collection_filter(store, embedder):
    load(store, embedder, doc="a", collection="alpha")
    load(store, embedder, doc="b", collection="beta")
    for hits in (
        store.search_dense(embedder.embed_query(TEXTS[0]), top_k=4, collection="beta"),
        store.search_keyword("ERR_4032", top_k=4, collection="beta"),
    ):
        assert hits and all(h.chunk_id.startswith("b:") for h in hits)


def test_reingesting_a_chunk_updates_it(store, embedder):
    load(store, embedder)
    load(store, embedder, texts=["Rewritten first chunk."])
    assert store.get("doc1:0").text == "Rewritten first chunk."
    assert store.stats()["chunks"] == len(TEXTS)


def test_delete_document_and_stats(store, embedder):
    load(store, embedder, doc="a")
    load(store, embedder, doc="b", collection="other")
    assert store.stats() == {"chunks": 8, "documents": 2, "collections": 2}
    assert store.delete_document("a") == 4
    assert store.delete_document("a") == 0
    assert store.stats() == {"chunks": 4, "documents": 1, "collections": 1}
    assert store.search_keyword("ERR_4032", top_k=5)[0].chunk_id == "b:0"


def test_rejects_wrong_vector_size(store):
    with pytest.raises(ValueError, match="64-dimensional"):
        store.add_chunks([chunk(0, "x")], [[0.1] * 8])


def test_refuses_to_start_on_a_dimension_mismatch(store):
    from app.store import PgVectorStore

    with pytest.raises(RuntimeError, match="vector\\(64\\)"):
        PgVectorStore(DATABASE_URL, 128)


def test_api_end_to_end_on_pgvector(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main
    from app.config import settings

    reset(DATABASE_URL)
    monkeypatch.setattr(settings, "store", "pgvector")
    monkeypatch.setattr(settings, "database_url", DATABASE_URL)
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))

    doc = "# Errors\n\n## ERR_4032\n\nERR_4032 means the connection pool is exhausted.\n"
    with TestClient(main.app) as client:
        assert client.post("/query", json={"question": "x"}).status_code == 409
        job = client.post(
            "/documents", files={"file": ("e.md", io.BytesIO(doc.encode()), "text/markdown")}
        ).json()
        deadline = time.time() + 10
        while client.get(f"/documents/{job['job_id']}").json()["status"] != "ready":
            assert time.time() < deadline, "ingestion did not finish"
            time.sleep(0.05)

        body = client.post("/query", json={"question": "ERR_4032", "generate": False}).json()
        assert body["retrieved"][0]["found_by_both"] is True
        assert "ERR_4032" in body["retrieved"][0]["text"]
        assert client.get("/health").json()["index"]["chunks"] >= 1
