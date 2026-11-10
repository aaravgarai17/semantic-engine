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
    def get_many(self, chunk_ids: Iterable[str]) -> list[StoredChunk]: ...
    def delete_document(self, document_id: str) -> int: ...
    def stats(self) -> dict: ...


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


class PgVectorStore:  # pragma: no cover - exercised by tests/test_pgvector.py against a live database
    """Postgres + pgvector, with HNSW for dense and tsvector for keyword search.

    Schema notes:

      * ``embedding vector(N)`` with an HNSW index using ``vector_cosine_ops``.
        Cosine rather than L2 because the embedders return normalised vectors,
        where cosine is the meaningful comparison.

      * ``tsv tsvector`` generated from the chunk text, with a GIN index.
        Keyword search here uses Postgres full-text search rather than the
        hand-written BM25 — ``ts_rank_cd`` is a different scoring function, so
        results differ slightly from the in-memory store. That is a deliberate
        trade of exact reproducibility for not shipping the whole corpus into
        the application process on every query.

    Every method is synchronous, matching ``InMemoryStore``, and draws a
    connection from a small pool. FastAPI runs sync endpoints in a thread pool,
    so a slow query blocks one worker thread, not the event loop.
    """

    def __init__(self, dsn: str, dimensions: int, pool_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        self.dsn = dsn
        self.dimensions = dimensions
        self._pool = ConnectionPool(dsn, min_size=1, max_size=pool_size, open=True)
        self.initialize()

    def close(self) -> None:
        self._pool.close()

    # ------------------------------------------------------------ setup

    def initialize(self) -> None:
        """Create the schema if needed and refuse a dimension mismatch.

        An existing table built for a different embedder would accept inserts
        of the wrong size only to fail on every query, so the mismatch is
        caught here, at startup, with a message that says what to do.
        """
        with self._pool.connection() as conn:
            conn.execute(self.schema_sql())
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone()
        if row and row[0] not in (-1, self.dimensions):
            raise RuntimeError(
                f"chunks.embedding is vector({row[0]}) but the embedder produces "
                f"{self.dimensions} dimensions; use a fresh database or re-ingest"
            )

    def schema_sql(self) -> str:
        return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    collection    TEXT NOT NULL DEFAULT 'default',
    title         TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection    TEXT NOT NULL DEFAULT 'default',
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embed_text    TEXT NOT NULL DEFAULT '',
    heading       TEXT NOT NULL DEFAULT '',
    page          INTEGER,
    embedding     vector({self.dimensions}) NOT NULL,
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

    # ------------------------------------------------------------ writes

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(x):.8g}" for x in vector) + "]"

    def add_chunks(
        self, chunks: Sequence[StoredChunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"expected {self.dimensions}-dimensional vectors, got {len(vector)}"
                )

        documents = {
            c.document_id: (c.document_id, c.collection, c.document_title) for c in chunks
        }
        rows = [
            (c.chunk_id, c.document_id, c.collection, c.index, c.text, c.embed_text,
             c.heading, c.page, self._vector_literal(v))
            for c, v in zip(chunks, vectors)
        ]

        # One transaction: a document's chunks appear all at once or not at all,
        # so a query never sees half an ingested document.
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (id, collection, title) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title",
                list(documents.values()),
            )
            cur.executemany(
                """INSERT INTO chunks
                       (id, document_id, collection, chunk_index, text, embed_text,
                        heading, page, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (id) DO UPDATE SET
                       text = EXCLUDED.text, embed_text = EXCLUDED.embed_text,
                       heading = EXCLUDED.heading, page = EXCLUDED.page,
                       embedding = EXCLUDED.embedding""",
                rows,
            )

    def delete_document(self, document_id: str) -> int:
        with self._pool.connection() as conn:
            removed = conn.execute(
                "DELETE FROM chunks WHERE document_id = %s RETURNING id", (document_id,)
            ).fetchall()
            conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        return len(removed)

    # ------------------------------------------------------------ search

    def search_dense(
        self, query_vector, top_k: int = 10, collection: Optional[str] = None
    ) -> list[ScoredChunk]:
        literal = self._vector_literal(query_vector)
        where, params = ("WHERE collection = %s", [collection]) if collection else ("", [])
        sql = f"""
            SELECT id, 1 - (embedding <=> %s::vector) AS score
            FROM chunks {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s"""
        with self._pool.connection() as conn:
            # HNSW applies a WHERE clause after the graph search, so a filtered
            # query can come back short. Widening the search beam (default 40)
            # keeps enough candidates for the filter to leave top_k behind.
            conn.execute(f"SET hnsw.ef_search = {max(40, top_k * 4)}")
            rows = conn.execute(sql, [literal, *params, literal, top_k]).fetchall()
        return [
            ScoredChunk(chunk_id=cid, score=float(score), rank=i + 1, source="dense")
            for i, (cid, score) in enumerate(rows)
        ]

    def search_keyword(
        self, query: str, top_k: int = 10, collection: Optional[str] = None
    ) -> list[ScoredChunk]:
        # plainto_tsquery ANDs every word, so a natural-language question would
        # only match chunks containing all of its terms. Build an OR query from
        # the question's own lexemes instead, which is what BM25 does, and let
        # ts_rank_cd reward chunks that match more of them. Each lexeme is
        # quoted so identifiers like libpq-dev or ERR_4032 aren't re-parsed.
        where = "AND c.collection = %s" if collection else ""
        sql = f"""
            WITH q AS (
                SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS query
                FROM unnest(to_tsvector('english', %s))
            )
            SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
            FROM chunks c, q
            WHERE q.query IS NOT NULL AND c.tsv @@ q.query {where}
            ORDER BY score DESC, c.id
            LIMIT %s"""
        params = [query, *([collection] if collection else []), top_k]
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ScoredChunk(chunk_id=cid, score=float(score), rank=i + 1, source="bm25")
            for i, (cid, score) in enumerate(rows)
        ]

    # ------------------------------------------------------------ reads

    _SELECT = """
        SELECT c.id, c.document_id, c.collection, c.text, c.embed_text, c.chunk_index,
               c.heading, c.page, d.title
        FROM chunks c JOIN documents d ON d.id = c.document_id"""

    @staticmethod
    def _row_to_chunk(row) -> StoredChunk:
        cid, doc_id, collection, text, embed_text, index, heading, page, title = row
        return StoredChunk(
            chunk_id=cid, document_id=doc_id, collection=collection, text=text,
            embed_text=embed_text, index=index, heading=heading or "", page=page,
            document_title=title or "",
        )

    def get(self, chunk_id: str) -> Optional[StoredChunk]:
        with self._pool.connection() as conn:
            row = conn.execute(self._SELECT + " WHERE c.id = %s", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_many(self, chunk_ids: Iterable[str]) -> list[StoredChunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(self._SELECT + " WHERE c.id = ANY(%s)", (ids,)).fetchall()
        by_id = {row[0]: self._row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in ids if cid in by_id]

    def stats(self) -> dict:
        with self._pool.connection() as conn:
            chunks, documents, collections = conn.execute(
                "SELECT count(*), count(DISTINCT document_id), count(DISTINCT collection) "
                "FROM chunks"
            ).fetchone()
        return {"chunks": chunks, "documents": documents, "collections": collections}


def build_store(kind: str, database_url: str, dimensions: int):
    """The store named by configuration: ``memory`` (default) or ``pgvector``."""
    kind = (kind or "memory").lower()
    if kind == "memory":
        return InMemoryStore()
    if kind == "pgvector":
        return PgVectorStore(database_url, dimensions)  # pragma: no cover - needs a database
    raise ValueError(f"unknown store {kind!r}; expected 'memory' or 'pgvector'")
