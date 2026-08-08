"""Pluggable embedding providers.

Why an interface rather than one provider
------------------------------------------
Three different consumers need different trade-offs:

  * **Tests** need something instant and deterministic with no dependencies.
  * **The evaluation** needs *real* semantics — the whole claim is that dense
    retrieval matches paraphrases, which a hash-based stand-in cannot do.
  * **Production** may want a hosted API for quality and to avoid shipping
    model weights.

`LocalEmbedder` is the default because it makes the headline result
reproducible: anyone can clone the repo and re-run the retrieval comparison
without an API key or a bill.

Batching
--------
Every implementation embeds in batches. One API call per chunk is the classic
mistake in an ingestion pipeline — a 300-page PDF is thousands of chunks, and
per-call latency dominates completely. Batching turns thousands of round trips
into tens.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Iterable, Protocol, Sequence

log = logging.getLogger("engine.embeddings")


class Embedder(Protocol):
    """Anything that turns text into vectors."""

    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _normalize(vector: list[float]) -> list[float]:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class HashEmbedder:
    """Deterministic pseudo-embeddings from hashed token n-grams.

    **Not semantic.** Two paraphrases with no shared tokens land far apart, so
    this cannot demonstrate anything about dense retrieval quality. It exists
    so unit tests run in milliseconds with no model download, and so the API
    can be exercised end to end offline.

    The evaluation deliberately refuses to run with this embedder — reporting
    recall figures from a non-semantic model would be actively misleading.
    """

    semantic = False

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()

        # Unigrams and bigrams, so word order carries a little signal.
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            slot = struct.unpack("<Q", digest)[0] % self.dimensions
            vector[slot] += 1.0

        return _normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class LocalEmbedder:
    """Sentence-transformers running locally. The default.

    `all-MiniLM-L6-v2` is ~80 MB and produces 384-dimensional vectors. It is
    substantially weaker than a large hosted model, which is worth stating
    plainly — but it is genuinely semantic, free, offline, and reproducible,
    which matters more for a result that is meant to be checked by others.
    """

    semantic = True

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "LocalEmbedder needs sentence-transformers:\n"
                "    pip install sentence-transformers\n"
                "Or use HashEmbedder for non-semantic offline testing."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VoyageEmbedder:  # pragma: no cover - requires an API key
    """Voyage AI, which Anthropic recommends for retrieval.

    Note the asymmetric `input_type`: documents and queries are embedded with
    different prefixes. Retrieval models are trained on that asymmetry, and
    ignoring it measurably costs recall — a subtle bug because everything still
    appears to work.
    """

    semantic = True

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        batch_size: int = 128,
    ) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1024

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            result = self._client.embed(batch, model=self.model, input_type="document")
            out.extend(result.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]


class OpenAIEmbedder:  # pragma: no cover - requires an API key
    semantic = True

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1536

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in response.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


def build_embedder(provider: str, api_key: str = "", model: str = "") -> Embedder:
    """Construct an embedder by name."""
    provider = provider.lower()

    if provider == "local":
        return LocalEmbedder(model or "all-MiniLM-L6-v2")
    if provider == "hash":
        return HashEmbedder()
    if provider == "voyage":
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for provider 'voyage'")
        return VoyageEmbedder(api_key, model or "voyage-3")
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for provider 'openai'")
        return OpenAIEmbedder(api_key, model or "text-embedding-3-small")

    raise ValueError(f"unknown embedding provider: {provider!r}")
