"""API endpoints, driven with the hash embedder so no model download is needed."""

import io
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    """Force the non-semantic embedder so tests need no model weights."""
    monkeypatch.setattr(settings, "embedding_provider", "hash")
    monkeypatch.setattr(settings, "anthropic_api_key", "")   # retrieval only
    monkeypatch.setattr(settings, "chunk_size", 400)
    monkeypatch.setattr(settings, "chunk_overlap", 50)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    with TestClient(main.app) as c:
        yield c


def upload(client, name: str, text: str, collection: str = "default"):
    return client.post(
        "/documents",
        files={"file": (name, io.BytesIO(text.encode()), "text/markdown")},
        data={"collection": collection},
    )


def wait_ready(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll until the job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/documents/{job_id}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


DOC = """# Error Reference

## ERR_4032

ERR_4032 is emitted when the connection pool is exhausted and no handle becomes
free within the wait threshold.

## Credential Recovery

If someone can no longer sign in, the system dispatches a single-use link to
the address held on file.
"""


# ------------------------------------------------------------------- health


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["embedder"] == "HashEmbedder"
    assert body["can_generate"] is False


# ------------------------------------------------------------------- upload


def test_upload_returns_202_immediately(client):
    """The work is not done when the response is sent — hence 202, not 200."""
    response = upload(client, "doc.md", DOC)

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["poll"] == f"/documents/{body['job_id']}"


def test_ingestion_completes_and_indexes(client):
    job_id = upload(client, "doc.md", DOC).json()["job_id"]
    body = wait_ready(client, job_id)

    assert body["status"] == "ready"
    assert body["chunks"] >= 1
    assert body["progress"] == 100

    assert client.get("/health").json()["index"]["chunks"] >= 1


def test_unsupported_file_type_rejected(client):
    response = client.post(
        "/documents",
        files={"file": ("image.png", io.BytesIO(b"\x89PNG"), "image/png")},
    )
    assert response.status_code == 415


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    response = upload(client, "big.md", "x" * 5000)
    assert response.status_code == 413


def test_unknown_job_returns_404(client):
    assert client.get("/documents/nosuchjob").status_code == 404


# -------------------------------------------------------------------- query


def test_query_before_any_documents_returns_409(client):
    response = client.post("/query", json={"question": "anything?"})
    assert response.status_code == 409


def test_hybrid_query_returns_ranked_chunks(client):
    wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    body = client.post("/query", json={"question": "ERR_4032", "generate": False}).json()

    assert body["method"] == "hybrid"
    assert body["retrieved"]
    assert "ERR_4032" in body["retrieved"][0]["text"]


def test_query_reports_which_retriever_found_each_chunk(client):
    """Provenance makes the fusion visible rather than a black box."""
    wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    body = client.post(
        "/query", json={"question": "connection pool exhausted", "generate": False}
    ).json()

    top = body["retrieved"][0]
    assert "dense_rank" in top and "bm25_rank" in top
    assert isinstance(top["found_by_both"], bool)


@pytest.mark.parametrize("method", ["hybrid", "dense", "bm25"])
def test_each_retrieval_method_works(client, method):
    wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    body = client.post(
        "/query", json={"question": "connection pool", "method": method, "generate": False}
    ).json()

    assert body["method"] == method
    assert body["retrieved"]


def test_invalid_method_rejected(client):
    response = client.post("/query", json={"question": "x", "method": "magic"})
    assert response.status_code == 422


def test_empty_question_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_top_k_is_respected(client):
    wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    body = client.post(
        "/query", json={"question": "the", "top_k": 1, "generate": False}
    ).json()
    assert len(body["retrieved"]) <= 1


def test_collection_scoping(client):
    wait_ready(client, upload(client, "a.md", DOC, collection="alpha").json()["job_id"])
    wait_ready(client, upload(client, "b.md", DOC, collection="beta").json()["job_id"])

    body = client.post(
        "/query",
        json={"question": "ERR_4032", "collection": "alpha", "generate": False},
    ).json()

    assert body["retrieved"]
    # Every hit must come from the requested collection.
    store = main.state["store"]
    for hit in body["retrieved"]:
        assert store.get(hit["chunk_id"]).collection == "alpha"


def test_query_without_api_key_returns_retrieval_only(client):
    wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    body = client.post("/query", json={"question": "ERR_4032"}).json()

    assert body["answer"] is None
    assert "ANTHROPIC_API_KEY" in body["note"]
    assert body["retrieved"]           # retrieval still works


# ------------------------------------------------------------------- delete


def test_delete_document(client):
    job = wait_ready(client, upload(client, "doc.md", DOC).json()["job_id"])

    response = client.delete(f"/documents/{job['document_id']}")
    assert response.status_code == 200
    assert response.json()["chunks_removed"] >= 1

    assert client.get("/health").json()["index"]["chunks"] == 0


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/documents/nosuch").status_code == 404
