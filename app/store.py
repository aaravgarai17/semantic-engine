"""Storage for chunks, vectors, and the keyword index.

Two implementations behind one interface:

  * ``InMemoryStore`` — dicts and brute-force cosine. Exact, dependency-free,
    and what the tests and the evaluation use. O(n) per query, which is fine
    for thousands of chunks and is *preferable* for evaluation because it
    introduces no approximation error to confuse the retrieval comparison.

  * ``PgVectorStore`` — Postgres with the pgvector extension and an HNSW
    index. What a deployment uses.

Exact versus approximate search
--------------------------------
Brute force compares the query against every vector: exact, and linear in
corpus size. HNSW builds a navigable graph and reaches a near neighbour in
roughly logarithmic time, at the cost of *occasionally missing a true nearest
neighbour*. That trade is right in production — 99% recall at 100× the speed —
but wrong for measuring retrieval quality, where an approximation error would
be indistinguishable from a ranking error.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence

from app.retrieval import BM25Index, ScoredChunk, dense_search

log = logging.getLogger("engine.store")


@dataclass
class StoredChunk:
    chunk_id: str
    document_id: str
    collection: str
    text: str
    embed_text: str
    index: int
    heading: str = ""
    page: Optional[int] = None
    document_title: str = ""

    def citation(self) -> str:
        bits = [self.document_title or self.document_id]
        if self.page is not None:
            bits.append(f"p.{self.page}")
        if self.heading:
            bits.append(self.heading)
        return " · ".join(bits)


class VectorStore(Protocol):
    def add_chunks(self, chunks: Sequence[StoredChunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def search_dense(self, query_vector, top_k, collection=None) -> list[ScoredChunk]: ...
    def search_keyword(self, query, top_k, collection=None) -> list[ScoredChunk]: ...
    def get(self, chunk_id: str) -> Optional[StoredChunk]: ...
    def delete_document(self, document_id: str) -> int: ...


class InMemoryStore:
    """Reference implementation. Exact search, no dependencies."""

    def __init__(self) -> None:
        self.chunks: dict[str, StoredChunk] = {}
        self.vectors: dict[str, list[float]] = {}
        self._bm25: Optional[BM25Index] = None
        self._bm25_stale = True

    def add_chunks(
        self, chunks: Sequence[StoredChunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        for chunk, vector in zip(chunks, vectors):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = list(vector)

        # BM25 statistics (document frequency, average length) are corpus-wide,
        # so adding documents invalidates them. Rebuilding lazily means a bulk
        # ingest of 50 documents rebuilds once rather than 50 times.
        self._bm25_stale = True

    def _keyword_index(self, collection: Optional[str]) -> BM25Index:
        if self._bm25 is None or self._bm25_stale:
            index = BM25Index()
            index.add_many(
                (cid, c.text) for cid, c in self.chunks.items()
            )
            self._bm25 = index
            self._bm25_stale = False
        return self._bm25

    def _filter(self, hits: list[ScoredChunk], collection: Optional[str]) -> list[ScoredChunk]:
        if collection is None:
            return hits
        return [
            h for h in hits
            if (c := self.chunks.get(h.chunk_id)) and c.collection == collection
        ]

    def search_dense(
        self, query_vector, top_k: int = 10, collection: Optional[str] = None
    ) -> list[ScoredChunk]:
        # Over-fetch when filtering, since post-filtering can otherwise leave
        # fewer than top_k results from the wrong collection.
        fetch = top_k * 4 if collection else top_k
        hits = dense_search(query_vector, self.vectors, top_k=fetch)
        return self._filter(hits, collection)[:top_k]

    def search_keyword(
        self, query: str, top_k: int = 10, collection: Optional[str] = None
    ) -> list[ScoredChunk]:
        fetch = top_k * 4 if collection else top_k
        hits = self._keyword_index(collection).search(query, top_k=fetch)
        return self._filter(hits, collection)[:top_k]

    def get(self, chunk_id: str) -> Optional[StoredChunk]:
        return self.chunks.get(chunk_id)

    def get_many(self, chunk_ids: Iterable[str]) -> list[StoredChunk]:
        return [c for cid in chunk_ids if (c := self.chunks.get(cid))]

    def delete_document(self, document_id: str) -> int:
        doomed = [cid for cid, c in self.chunks.items() if c.document_id == document_id]
        for cid in doomed:
            self.chunks.pop(cid, None)
            self.vectors.pop(cid, None)
        self._bm25_stale = True
        return len(doomed)

    def stats(self) -> dict:
        collections = {c.collection for c in self.chunks.values()}
        documents = {c.document_id for c in self.chunks.values()}
        return {
            "chunks": len(self.chunks),
            "documents": len(documents),
            "collections": len(collections),
        }


class PgVectorStore:  # pragma: no cover - requires a live database
    """Postgres + pgvector, with HNSW for dense and tsvector for keyword search.

    Schema notes:

      * ``embedding vector(N)`` with an HNSW index using ``vector_cosine_ops``.
        Cosine rather than L2 because the embedders return normalised vectors,
        where cosine is the meaningful comparison.

      * ``tsv tsvector`` maintained by a trigger, with a GIN index. Keyword
        search here uses Postgres full-text search rather than the hand-written
        BM25 — ``ts_rank_cd`` is a different scoring function, so results differ
        slightly from the in-memory store. That is a deliberate trade of exact
        reproducibility for not shipping the whole corpus into the application
        process on every query.
    """

    def __init__(self, dsn: str, dimensions: int) -> None:
        self.dsn = dsn
        self.dimensions = dimensions

    def schema_sql(self) -> str:
        return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    collection    TEXT NOT NULL DEFAULT 'default',
    title         TEXT NOT NULL,
    filename      TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection    TEXT NOT NULL DEFAULT 'default',
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    heading       TEXT,
    page          INTEGER,
    embedding     vector({self.dimensions}),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

-- Approximate nearest neighbour. m and ef_construction trade build time and
-- memory for recall; these are pgvector's defaults and are sane starting points.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_collection ON chunks (collection);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id);
"""
