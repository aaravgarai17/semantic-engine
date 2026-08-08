"""Store, ingestion, extraction, and the end-to-end retrieval path."""

import asyncio

import pytest

from app.embeddings import HashEmbedder
from app.extract import ExtractionError, extract
from app.ingest import Ingestor, Stage
from app.retrieval import reciprocal_rank_fusion
from app.store import InMemoryStore, StoredChunk


@pytest.fixture
def embedder():
    return HashEmbedder(dimensions=64)


@pytest.fixture
def store():
    return InMemoryStore()


def make_chunk(n: int, text: str, doc="doc1", collection="default") -> StoredChunk:
    return StoredChunk(
        chunk_id=f"{doc}:{n}",
        document_id=doc,
        collection=collection,
        text=text,
        embed_text=text,
        index=n,
        document_title=doc,
    )


# --------------------------------------------------------------- embeddings


def test_hash_embedder_is_deterministic(embedder):
    assert embedder.embed_query("hello world") == embedder.embed_query("hello world")


def test_hash_embedder_normalises(embedder):
    vector = embedder.embed_query("some text here")
    magnitude = sum(x * x for x in vector) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-6)


def test_hash_embedder_batches(embedder):
    vectors = embedder.embed(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 64 for v in vectors)


def test_hash_embedder_handles_empty_batch(embedder):
    assert embedder.embed([]) == []


def test_hash_embedder_is_marked_non_semantic():
    """Guards the eval's refusal to report dense numbers from it."""
    assert HashEmbedder().semantic is False


# -------------------------------------------------------------------- store


def test_add_and_retrieve(store, embedder):
    chunks = [make_chunk(0, "the database connection pool")]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))

    assert store.get("doc1:0").text == "the database connection pool"
    assert store.stats()["chunks"] == 1


def test_mismatched_lengths_raise(store):
    with pytest.raises(ValueError):
        store.add_chunks([make_chunk(0, "x")], [])


def test_keyword_search(store, embedder):
    chunks = [
        make_chunk(0, "database connection pooling explained"),
        make_chunk(1, "invoice generation runs monthly"),
    ]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))

    hits = store.search_keyword("database pooling", top_k=5)
    assert hits[0].chunk_id == "doc1:0"


def test_dense_search(store, embedder):
    chunks = [make_chunk(0, "alpha beta"), make_chunk(1, "gamma delta")]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))

    hits = store.search_dense(embedder.embed_query("alpha beta"), top_k=2)
    assert hits[0].chunk_id == "doc1:0"


def test_collection_filtering(store, embedder):
    chunks = [
        make_chunk(0, "shared keyword here", doc="a", collection="alpha"),
        make_chunk(0, "shared keyword here", doc="b", collection="beta"),
    ]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))

    hits = store.search_keyword("shared keyword", top_k=10, collection="alpha")
    assert [h.chunk_id for h in hits] == ["a:0"]


def test_keyword_index_rebuilds_after_new_documents(store, embedder):
    """BM25 statistics are corpus-wide, so adding documents invalidates them."""
    first = [make_chunk(0, "original content")]
    store.add_chunks(first, embedder.embed([c.embed_text for c in first]))
    assert len(store.search_keyword("original", top_k=5)) == 1

    second = [make_chunk(1, "original content again", doc="doc2")]
    store.add_chunks(second, embedder.embed([c.embed_text for c in second]))

    assert len(store.search_keyword("original", top_k=5)) == 2


def test_delete_document_removes_all_its_chunks(store, embedder):
    chunks = [make_chunk(i, f"content {i}") for i in range(3)]
    chunks.append(make_chunk(0, "other doc", doc="doc2"))
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))

    assert store.delete_document("doc1") == 3
    assert store.stats()["chunks"] == 1
    assert store.get("doc1:0") is None


def test_delete_updates_the_keyword_index(store, embedder):
    chunks = [make_chunk(0, "findable content")]
    store.add_chunks(chunks, embedder.embed([c.embed_text for c in chunks]))
    store.delete_document("doc1")

    assert store.search_keyword("findable", top_k=5) == []


def test_citation_string(store):
    chunk = StoredChunk(
        chunk_id="x", document_id="d", collection="c", text="t", embed_text="t",
        index=0, heading="Setup > Linux", page=12, document_title="Manual",
    )
    assert chunk.citation() == "Manual · p.12 · Setup > Linux"


# ---------------------------------------------------------------- extraction


def test_extract_plain_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Hello world.\n\nSecond paragraph.")

    document = extract(path)
    assert "Hello world" in document.text
    assert document.format == "txt"


def test_extract_markdown(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody text.")

    assert "# Title" in extract(path).text


def test_extract_normalises_whitespace(tmp_path):
    """Runs of blank lines would confuse the chunker's paragraph detection."""
    path = tmp_path / "messy.txt"
    path.write_text("Line one.\n\n\n\n\nLine two.\x0c\xa0More.")

    text = extract(path).text
    assert "\n\n\n" not in text
    assert "\xa0" not in text


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")

    with pytest.raises(ExtractionError, match="unsupported"):
        extract(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ExtractionError, match="not found"):
        extract(tmp_path / "nope.txt")


# ---------------------------------------------------------------- ingestion


async def test_ingests_a_document(tmp_path, store, embedder):
    path = tmp_path / "guide.md"
    path.write_text(
        "# Setup\n\n" + "Installation requires several steps to complete. " * 20
        + "\n\n## Linux\n\n" + "On Debian you need extra packages installed. " * 20
    )

    ingestor = Ingestor(store, embedder, chunk_size=400, overlap=50)
    job = ingestor.submit(path, collection="docs")
    await ingestor.run(job, path)

    assert job.stage is Stage.READY
    assert job.chunk_count > 1
    assert job.error is None
    assert store.stats()["chunks"] == job.chunk_count


async def test_ingested_chunks_are_searchable(tmp_path, store, embedder):
    path = tmp_path / "errors.md"
    path.write_text(
        "# Errors\n\n" + "The ERR_4032 code means the connection pool was exhausted. " * 10
    )

    ingestor = Ingestor(store, embedder, chunk_size=500, overlap=50)
    job = ingestor.submit(path)
    await ingestor.run(job, path)

    hits = store.search_keyword("ERR_4032", top_k=5)
    assert hits
    assert "ERR_4032" in store.get(hits[0].chunk_id).text


async def test_job_reports_progress_and_terminal_state(tmp_path, store, embedder):
    path = tmp_path / "d.txt"
    path.write_text("Some content here. " * 100)

    ingestor = Ingestor(store, embedder)
    job = ingestor.submit(path)
    assert job.progress == 0

    await ingestor.run(job, path)
    assert job.progress == 100
    assert job.stage.is_terminal


async def test_failure_is_recorded_not_raised(tmp_path, store, embedder):
    """A failed ingestion must leave a readable status, not crash the worker."""
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ")

    ingestor = Ingestor(store, embedder)
    job = ingestor.submit(path)
    await ingestor.run(job, path)

    assert job.stage is Stage.FAILED
    assert job.error
    assert job.stage.is_terminal


async def test_jobs_are_retrievable_by_id(tmp_path, store, embedder):
    path = tmp_path / "d.txt"
    path.write_text("content " * 50)

    ingestor = Ingestor(store, embedder)
    job = ingestor.submit(path)

    assert ingestor.get(job.job_id) is job
    assert ingestor.get("nonexistent") is None


async def test_concurrent_ingestions_are_bounded(tmp_path, store, embedder):
    """Fifty simultaneous embedding batches would exhaust memory or rate limits."""
    paths = []
    for i in range(6):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"Document {i} content. " * 60)
        paths.append(p)

    ingestor = Ingestor(store, embedder, max_concurrent=2)
    jobs = [ingestor.submit(p) for p in paths]

    await asyncio.gather(*(ingestor.run(j, p) for j, p in zip(jobs, paths)))

    assert all(j.stage is Stage.READY for j in jobs)
    assert store.stats()["documents"] == 6


# ------------------------------------------------------------- end to end


async def test_hybrid_retrieval_over_ingested_documents(tmp_path, store, embedder):
    """The full path: ingest, then retrieve with both methods fused."""
    (tmp_path / "a.md").write_text(
        "# Auth\n\nCredential recovery sends a single-use link to the address on file."
    )
    (tmp_path / "b.md").write_text(
        "# Errors\n\nERR_4032 indicates the connection pool has been exhausted."
    )

    ingestor = Ingestor(store, embedder, chunk_size=1000, overlap=0)
    for name in ("a.md", "b.md"):
        path = tmp_path / name
        job = ingestor.submit(path)
        await ingestor.run(job, path)

    dense = store.search_dense(embedder.embed_query("ERR_4032"), top_k=10)
    keyword = store.search_keyword("ERR_4032", top_k=10)
    fused = reciprocal_rank_fusion([dense, keyword], top_k=5)

    assert fused
    assert "ERR_4032" in store.get(fused[0].chunk_id).text
