"""FastAPI application.

Upload returns 202 immediately and ingests in the background; query runs the
hybrid retrieval and generates a cited answer.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings import build_embedder
from app.extract import SUPPORTED
from app.generate import answer_question
from app.ingest import Ingestor
from app.retrieval import reciprocal_rank_fusion
from app.cache import build_cache, cache_key
from app.store import VectorStore, build_store

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("engine")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = settings.voyage_api_key or settings.openai_api_key
    embedder = build_embedder(
        settings.embedding_provider, api_key, settings.embedding_model
    )

    store = build_store(settings.store, settings.database_url, embedder.dimensions)
    cache = build_cache(settings.redis_url, settings.cache_ttl_seconds)
    state["embedder"] = embedder
    state["store"] = store
    state["cache"] = cache
    state["ingestor"] = Ingestor(
        store,
        embedder,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        max_concurrent=settings.max_concurrent_ingestions,
        embed_batch=settings.embed_batch_size,
        on_indexed=cache.invalidate,
    )

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    log.info(
        "ready: %s embedder (%dd), %s store, %s",
        type(embedder).__name__, embedder.dimensions, settings.store,
        type(cache).__name__,
    )
    if not settings.can_generate:
        log.warning("ANTHROPIC_API_KEY not set — /query will retrieve but not generate")

    yield

    if hasattr(store, "close"):
        store.close()


app = FastAPI(
    title="Semantic Document Engine",
    description="Hybrid retrieval (dense + BM25 fused with RRF) over your documents, "
                "with cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- schemas


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: Optional[int] = None
    generate: bool = True
    method: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    citation: str
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    found_by_both: bool = False


# ---------------------------------------------------------------- endpoints


@app.get("/health")
def health():
    store = state.get("store")
    embedder = state.get("embedder")
    return {
        "status": "ok",
        "embedder": type(embedder).__name__ if embedder else None,
        "dimensions": getattr(embedder, "dimensions", None),
        "can_generate": settings.can_generate,
        "index": store.stats() if store else {},
    }


@app.post("/documents", status_code=202)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form("default"),
):
    """Accept a document and ingest it in the background.

    Returns 202 rather than 200 because the work is not done — a 300-page PDF
    takes tens of seconds, far past any sensible request timeout.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported type {suffix!r}; supported: {sorted(SUPPORTED)}",
        )

    destination = Path(settings.upload_dir) / f"{uuid.uuid4().hex[:12]}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    with destination.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds {settings.max_upload_mb} MB",
                )
            out.write(chunk)

    ingestor: Ingestor = state["ingestor"]
    job = ingestor.submit(destination, collection=collection, title=file.filename or "")
    background.add_task(ingestor.run, job, destination)

    return {**job.to_dict(), "poll": f"/documents/{job.job_id}"}


@app.get("/documents/{job_id}")
def job_status(job_id: str):
    job = state["ingestor"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.to_dict()


@app.post("/query")
def query(request: QueryRequest):
    """Retrieve and, if configured, generate a cited answer."""
    store: VectorStore = state["store"]
    embedder = state["embedder"]
    cache = state["cache"]
    top_k = request.top_k or settings.top_k
    will_generate = request.generate and settings.can_generate

    # Everything that can change the response goes into the key, so a hit is
    # always the answer this exact request would have produced.
    generation = cache.generation()
    key = cache_key(
        generation,
        question=request.question, collection=request.collection, top_k=top_k,
        method=request.method, generate=will_generate, model=settings.model,
        candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
        min_score=settings.min_retrieval_score,
        max_context_chars=settings.max_context_chars,
        max_tokens=settings.max_answer_tokens,
    )
    if generation >= 0 and (hit := cache.get(key)) is not None:
        return {**hit, "cached": True}

    if store.stats()["chunks"] == 0:
        raise HTTPException(status_code=409, detail="no documents have been indexed yet")

    dense_hits = bm25_hits = []
    if request.method in ("hybrid", "dense"):
        vector = embedder.embed_query(request.question)
        dense_hits = store.search_dense(
            vector, top_k=settings.candidate_k, collection=request.collection
        )
    if request.method in ("hybrid", "bm25"):
        bm25_hits = store.search_keyword(
            request.question, top_k=settings.candidate_k, collection=request.collection
        )

    if request.method == "hybrid":
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits], k=settings.rrf_k, top_k=top_k
        )
        ranked = [(f.chunk_id, f.score, f.dense_rank, f.bm25_rank) for f in fused]
    else:
        hits = dense_hits if request.method == "dense" else bm25_hits
        ranked = [(h.chunk_id, h.score, None, None) for h in hits[:top_k]]

    chunks = []
    payload = []
    for chunk_id, score, d_rank, b_rank in ranked:
        chunk = store.get(chunk_id)
        if chunk is None:
            continue
        chunks.append(chunk)
        payload.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.text,
                score=round(score, 6),
                citation=chunk.citation(),
                dense_rank=d_rank,
                bm25_rank=b_rank,
                found_by_both=d_rank is not None and b_rank is not None,
            ).model_dump()
        )

    response = {
        "question": request.question,
        "method": request.method,
        "retrieved": payload,
    }

    if not will_generate:
        response["answer"] = None
        response["note"] = (
            "retrieval only — set ANTHROPIC_API_KEY to generate answers"
            if not settings.can_generate else "generation not requested"
        )
        return _cache_and_return(cache, generation, key, response)

    from app.llm import ClaudeLLM

    answer = answer_question(
        ClaudeLLM(settings.anthropic_api_key, settings.model),
        request.question,
        chunks,
        scores=[s for _, s, _, _ in ranked],
        min_score=settings.min_retrieval_score,
        max_context_chars=settings.max_context_chars,
        max_tokens=settings.max_answer_tokens,
    )

    response["answer"] = {
        "text": answer.text,
        "refused": answer.refused,
        "grounded": answer.is_grounded,
        "citations": [
            {
                "number": c.number,
                "label": c.label(),
                "chunk_id": c.chunk_id,
                "excerpt": c.excerpt,
            }
            for c in answer.citations
        ],
        "tokens": {"input": answer.input_tokens, "output": answer.output_tokens},
    }
    return _cache_and_return(cache, generation, key, response)


def _cache_and_return(cache, generation: int, key: str, response: dict) -> dict:
    # A negative generation means the cache was unreachable when the request
    # started; storing under it could serve this answer after a later outage.
    if generation >= 0:
        cache.set(key, response)
    return {**response, "cached": False}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    removed = state["store"].delete_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="unknown document")
    state["cache"].invalidate()
    return {"document_id": document_id, "chunks_removed": removed}
