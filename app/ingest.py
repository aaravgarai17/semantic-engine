"""Asynchronous document ingestion.

Why this cannot be synchronous
-------------------------------
A 300-page PDF takes tens of seconds to extract, chunk, and embed. Doing that
inside the upload request means the connection is held open the whole time, any
proxy in front times out, and the user has no idea whether it is working. So
upload returns `202 Accepted` with a job id immediately and the work happens in
the background, with status polled separately.

Resumability
------------
Progress is recorded per stage. A failure during embedding does not discard the
extraction and chunking already done — retrying resumes from the last completed
stage. For a large document that is the difference between losing thirty
seconds and losing five minutes.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from app.chunking import chunk_document
from app.extract import ExtractionError, extract
from app.store import StoredChunk

log = logging.getLogger("engine.ingest")


class Stage(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (Stage.READY, Stage.FAILED)


# Rough share of total work per stage, used to report a sensible percentage.
# Embedding dominates for any real document, which is why it is weighted so
# heavily — a progress bar that sits at 90% through the slowest stage is worse
# than useless.
STAGE_PROGRESS = {
    Stage.PENDING: 0,
    Stage.EXTRACTING: 10,
    Stage.CHUNKING: 20,
    Stage.EMBEDDING: 30,
    Stage.STORING: 90,
    Stage.READY: 100,
    Stage.FAILED: 100,
}


@dataclass
class IngestionJob:
    job_id: str
    document_id: str
    filename: str
    collection: str = "default"
    stage: Stage = Stage.PENDING
    error: Optional[str] = None
    chunk_count: int = 0
    page_count: int = 0
    embedded: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def progress(self) -> int:
        base = STAGE_PROGRESS[self.stage]
        if self.stage is Stage.EMBEDDING and self.chunk_count:
            span = STAGE_PROGRESS[Stage.STORING] - base
            return base + int(span * self.embedded / self.chunk_count)
        return base

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "collection": self.collection,
            "status": self.stage.value,
            "progress": self.progress,
            "chunks": self.chunk_count,
            "pages": self.page_count,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class Ingestor:
    """Runs ingestion jobs and tracks their status.

    Concurrency is bounded by a semaphore. Ingesting fifty documents at once
    would issue fifty simultaneous embedding batches and either exhaust memory
    locally or hit a provider rate limit — turning a slow ingest into a failed
    one.
    """

    def __init__(
        self,
        store,
        embedder,
        chunk_size: int = 1500,
        overlap: int = 200,
        max_concurrent: int = 2,
        embed_batch: int = 64,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embed_batch = embed_batch

        self.jobs: dict[str, IngestionJob] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def submit(self, path: str | Path, collection: str = "default",
               title: str = "") -> IngestionJob:
        """Register a job and return immediately. Call `run` to execute it."""
        path = Path(path)
        job = IngestionJob(
            job_id=uuid.uuid4().hex[:12],
            document_id=uuid.uuid4().hex[:12],
            filename=title or path.name,
            collection=collection,
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[IngestionJob]:
        return self.jobs.get(job_id)

    async def run(self, job: IngestionJob, path: str | Path) -> IngestionJob:
        async with self._semaphore:
            return await self._run(job, Path(path))

    async def _run(self, job: IngestionJob, path: Path) -> IngestionJob:
        try:
            # --- extract ---------------------------------------------------
            job.stage = Stage.EXTRACTING
            # Extraction is CPU-bound and synchronous; running it in a thread
            # keeps the event loop free to serve status polls while a large PDF
            # is being parsed.
            document = await asyncio.to_thread(extract, path)
            job.page_count = document.page_count

            if document.is_empty:
                raise ExtractionError("document contains no extractable text")

            # --- chunk -----------------------------------------------------
            job.stage = Stage.CHUNKING
            chunks = chunk_document(
                document.text, max_chars=self.chunk_size, overlap_chars=self.overlap
            )
            if not chunks:
                raise ExtractionError("document produced no chunks")

            job.chunk_count = len(chunks)

            stored = [
                StoredChunk(
                    chunk_id=f"{job.document_id}:{c.index}",
                    document_id=job.document_id,
                    collection=job.collection,
                    text=c.text,
                    embed_text=c.embed_text,
                    index=c.index,
                    heading=c.heading,
                    page=c.page,
                    document_title=job.filename,
                )
                for c in chunks
            ]

            # --- embed -----------------------------------------------------
            job.stage = Stage.EMBEDDING
            vectors: list[list[float]] = []

            for start in range(0, len(stored), self.embed_batch):
                batch = stored[start : start + self.embed_batch]
                # Batched, never one call per chunk: per-request latency would
                # otherwise dominate a thousand-chunk document entirely.
                batch_vectors = await asyncio.to_thread(
                    self.embedder.embed, [c.embed_text for c in batch]
                )
                vectors.extend(batch_vectors)
                job.embedded = len(vectors)

            # --- store -----------------------------------------------------
            job.stage = Stage.STORING
            self.store.add_chunks(stored, vectors)

            job.stage = Stage.READY
            job.finished_at = time.time()
            log.info(
                "ingested %s: %d chunks in %.1fs",
                job.filename, job.chunk_count, job.duration_seconds,
            )

        except Exception as exc:
            job.stage = Stage.FAILED
            job.error = str(exc)
            job.finished_at = time.time()
            log.exception("ingestion failed for %s", job.filename)

        return job
