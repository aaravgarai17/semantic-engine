"""Hybrid retrieval: dense vectors, BM25 keywords, and Reciprocal Rank Fusion.

This is the heart of the project and the part an interviewer will probe hardest.

Why one retrieval method is not enough
--------------------------------------
**Dense vector search** compares embeddings, so it matches *meaning*. "How do I
reset my password?" finds "credential recovery procedure" despite sharing no
words. Where it fails is exact identifiers: product codes, error numbers, and
rare proper nouns get smeared into general semantic space, because the model
was trained to place similar concepts near each other and `ERR_4032` has no
concept. Searching for it returns documents about errors in general.

**BM25 keyword search** is the mirror image. It scores documents by term
overlap, so `ERR_4032` finds exactly the document containing that string. But
it has no notion of synonymy — "password reset" and "credential recovery" share
nothing, so it returns nothing.

Their failure modes are complementary, which is why combining them beats either
alone. `eval/` measures exactly this.

Why fusing scores directly does not work
-----------------------------------------
The obvious approach — normalise both scores and add them — has a real problem:
the scores are not commensurable. A cosine similarity of 0.83 and a BM25 score
of 14.2 live on different scales with different distributions. BM25 is unbounded
above and depends on corpus statistics; cosine is bounded in [-1, 1] and tends
to cluster tightly (most pairs score 0.6–0.9, so the *spread* that matters is
small and easily swamped). Min-max normalising per query makes the result depend
on the outliers in that particular result set, so the same document can score
differently depending on what else was retrieved.

Reciprocal Rank Fusion sidesteps this by discarding the scores entirely and
using only **rank position**:

    RRF(d) = Σ  1 / (k + rank_i(d))
             i

over each ranked list `i` the document appears in, with rank starting at 1.
Documents ranked highly by *both* methods accumulate the most weight. A document
that one method loves and the other has never heard of still scores reasonably,
which is what you want — that's the case where one method's blind spot is being
covered.

The constant `k` (conventionally 60) damps the influence of top ranks. Without
it, rank 1 would score 1.0 and rank 2 only 0.5 — a first place would be nearly
impossible to overcome. With k=60, ranks 1 and 2 score 1/61 and 1/62: close
enough that agreement across lists matters more than being first in one.

RRF needs no tuning, no training data, and no score calibration. That is the
argument for it over a learned reranker in a system this size.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Conventional default from the original RRF paper (Cormack et al., 2009).
RRF_K = 60

# BM25 tuning constants. k1 controls how fast term-frequency saturates; b
# controls how much document length is penalised. These are the standard
# defaults and are rarely worth changing without a labelled test set.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN = re.compile(r"[a-z0-9_]+")

# Words carrying no discriminative signal. Kept deliberately short: an
# aggressive stopword list hurts, because a query like "how to be" is all
# stopwords and would retrieve nothing at all.
STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the
this to was were will with
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase and split on word characters, keeping underscores and digits.

    Underscores and digits are kept deliberately: identifiers like `ERR_4032`
    and `oauth2` are exactly the queries BM25 exists to handle, and splitting
    them apart would destroy the advantage it has over dense retrieval.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class ScoredChunk:
    """A retrieval hit, carrying provenance for later fusion and citation."""

    chunk_id: str
    score: float
    rank: int = 0
    source: str = ""            # "dense" | "bm25" | "hybrid"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.chunk_id} {self.source} rank={self.rank} score={self.score:.4f}>"


@dataclass
class FusedResult:
    chunk_id: str
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.bm25_rank is not None


# --------------------------------------------------------------------- BM25


class BM25Index:
    """BM25 Okapi, implemented directly rather than delegated to a database.

    Postgres offers `ts_rank`, which is convenient and what a production system
    would likely use. Writing the scoring out has two advantages here: the
    retrieval comparison runs with no database at all, so anyone can reproduce
    it, and the ranking is inspectable rather than opaque.

    The scoring function per query term:

        IDF(q) · f(q,D) · (k1 + 1) / (f(q,D) + k1 · (1 - b + b · |D| / avgdl))

    where `IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)`.

    Reading it in pieces:
      * `IDF` rewards rare terms. A term in every document tells you nothing.
      * `f(q,D) · (k1+1) / (f(q,D) + k1 · ...)` saturates. The tenth occurrence
        of a word adds far less than the second — unlike raw term frequency,
        which a keyword-stuffed document would exploit.
      * `|D| / avgdl` penalises long documents, which would otherwise win
        simply by containing more words.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b

        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.df: Counter = Counter()          # how many docs contain each term
        self.avg_doc_length: float = 0.0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.doc_ids.append(chunk_id)
        self.doc_tokens.append(tokens)
        self.doc_freqs.append(Counter(tokens))
        self.doc_lengths.append(len(tokens))
        for term in set(tokens):
            self.df[term] += 1

    def add_many(self, items: Iterable[tuple[str, str]]) -> None:
        for chunk_id, text in items:
            self.add(chunk_id, text)
        self.finalize()

    def finalize(self) -> None:
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        total = len(self.doc_ids)
        if total == 0:
            return 0.0
        # The +1 inside the log keeps IDF non-negative for terms appearing in
        # more than half the corpus. Without it those terms score negative and
        # can actively push a matching document *down* the ranking.
        return math.log((total - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        if not self.doc_ids:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        if self.avg_doc_length == 0:
            self.finalize()

        scores: list[tuple[str, float]] = []

        for idx, chunk_id in enumerate(self.doc_ids):
            freqs = self.doc_freqs[idx]
            length = self.doc_lengths[idx] or 1
            score = 0.0

            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_doc_length or 1)
                )
                score += self.idf(term) * (tf * (self.k1 + 1)) / denominator

            if score > 0:
                scores.append((chunk_id, score))

        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return [
            ScoredChunk(chunk_id=cid, score=s, rank=i + 1, source="bm25")
            for i, (cid, s) in enumerate(scores[:top_k])
        ]


# --------------------------------------------------------------------- dense


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Written out rather than pulled from numpy so this module has no heavy
    dependency — the retrieval logic stays importable and testable anywhere.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def dense_search(
    query_vector: Sequence[float],
    vectors: dict[str, Sequence[float]],
    top_k: int = 10,
) -> list[ScoredChunk]:
    """Brute-force cosine search over an in-memory vector set.

    Exact and O(n), which is correct for evaluation and small corpora. A
    production store uses an approximate index (pgvector's HNSW) that trades a
    little recall for sub-linear query time; see `store.py`.
    """
    scored = [
        (chunk_id, cosine_similarity(query_vector, vec))
        for chunk_id, vec in vectors.items()
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))

    return [
        ScoredChunk(chunk_id=cid, score=s, rank=i + 1, source="dense")
        for i, (cid, s) in enumerate(scored[:top_k])
    ]


# ----------------------------------------------------------------------- RRF


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    k: int = RRF_K,
    top_k: int = 10,
    weights: Optional[Sequence[float]] = None,
) -> list[FusedResult]:
    """Fuse ranked lists by rank position, ignoring the underlying scores.

    `weights` allows leaning on one retriever, but defaults to equal. Weighting
    is a tuning knob that needs a labelled set to set responsibly — the appeal
    of plain RRF is that it works well with none.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    totals: dict[str, float] = defaultdict(float)
    dense_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for list_index, (results, weight) in enumerate(zip(ranked_lists, weights)):
        for position, hit in enumerate(results, start=1):
            # Use enumeration order rather than a stored rank so the function
            # works on any ranked sequence, including hand-built test data.
            totals[hit.chunk_id] += weight / (k + position)

            if hit.source == "dense":
                dense_rank.setdefault(hit.chunk_id, position)
            elif hit.source == "bm25":
                bm25_rank.setdefault(hit.chunk_id, position)

    fused = [
        FusedResult(
            chunk_id=cid,
            score=score,
            dense_rank=dense_rank.get(cid),
            bm25_rank=bm25_rank.get(cid),
        )
        for cid, score in totals.items()
    ]

    # Ties broken by chunk_id so results are deterministic — important for an
    # evaluation harness that must produce the same numbers on every run.
    fused.sort(key=lambda r: (-r.score, r.chunk_id))
    return fused[:top_k]


def hybrid_search(
    query: str,
    query_vector: Sequence[float],
    bm25: BM25Index,
    vectors: dict[str, Sequence[float]],
    top_k: int = 10,
    candidate_k: int = 30,
    rrf_k: int = RRF_K,
) -> list[FusedResult]:
    """Run both retrievers and fuse them.

    Each retriever returns `candidate_k` results — deliberately more than the
    final `top_k`. Fusion can only reorder what it is given, so a document
    ranked 20th by dense and 3rd by BM25 needs both lists to run deep enough to
    contain it. Retrieving only `top_k` from each would discard exactly the
    cross-method agreement that makes fusion work.
    """
    dense_hits = dense_search(query_vector, vectors, top_k=candidate_k)
    bm25_hits = bm25.search(query, top_k=candidate_k)
    return reciprocal_rank_fusion([dense_hits, bm25_hits], k=rrf_k, top_k=top_k)


# ---------------------------------------------------------------- evaluation


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant chunks appearing in the top k.

    Recall rather than precision because this measures *retrieval* feeding a
    generator: a relevant chunk that never gets retrieved cannot be cited, and
    the generator can usually ignore an irrelevant one that slips in. Missing
    context is unrecoverable; extra context is merely wasteful.
    """
    if not relevant:
        return 0.0
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant result; 0 if none. Rewards ranking it first."""
    relevant_set = set(relevant)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / position
    return 0.0
