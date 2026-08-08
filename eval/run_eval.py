"""Retrieval evaluation: does hybrid actually beat dense and BM25 alone?

This produces the single most important artifact in the project — a table
comparing recall@k for each retrieval strategy, broken down by question type so
the *reason* hybrid wins is visible rather than asserted.

Run:  python -m eval.run_eval
      python -m eval.run_eval --embedder hash     (fast, non-semantic, for CI)
      python -m eval.run_eval --chunk-size 512    (chunk-size experiment)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

from app.chunking import chunk_document
from app.embeddings import HashEmbedder, build_embedder
from app.retrieval import (
    BM25Index,
    dense_search,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank_fusion,
)
from eval.dataset import DOCUMENTS, build_questions


@dataclass
class Scores:
    recall_5: list[float] = field(default_factory=list)
    recall_10: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    def add(self, retrieved: list[str], relevant: list[str], elapsed_ms: float) -> None:
        self.recall_5.append(recall_at_k(retrieved, relevant, 5))
        self.recall_10.append(recall_at_k(retrieved, relevant, 10))
        self.mrr.append(mean_reciprocal_rank(retrieved, relevant))
        self.latency_ms.append(elapsed_ms)

    def mean(self, attr: str) -> float:
        values = getattr(self, attr)
        return statistics.mean(values) if values else 0.0


def build_corpus(chunk_size: int, overlap: int):
    """Chunk every document and return the pieces plus lookup tables."""
    chunks_by_id: dict[str, str] = {}
    embed_by_id: dict[str, str] = {}

    for doc in DOCUMENTS:
        for chunk in chunk_document(doc.text, max_chars=chunk_size, overlap_chars=overlap):
            chunk_id = f"{doc.doc_id}:{chunk.index}"
            chunks_by_id[chunk_id] = chunk.text
            # Headings are prepended for embedding but not for BM25, which
            # already matches on the literal words present in the body.
            embed_by_id[chunk_id] = chunk.embed_text

    return chunks_by_id, embed_by_id


def evaluate(
    embedder,
    chunk_size: int,
    overlap: int,
    candidate_k: int = 30,
    weights: tuple[float, float] = (1.0, 1.0),
):
    chunks_by_id, embed_by_id = build_corpus(chunk_size, overlap)
    questions = build_questions(chunks_by_id)

    # --- index -------------------------------------------------------------
    bm25 = BM25Index()
    bm25.add_many(chunks_by_id.items())

    ids = list(embed_by_id.keys())
    t0 = time.perf_counter()
    vectors = dict(zip(ids, embedder.embed([embed_by_id[i] for i in ids])))
    index_seconds = time.perf_counter() - t0

    # --- run ---------------------------------------------------------------
    overall = {"dense": Scores(), "bm25": Scores(), "hybrid": Scores()}
    by_kind = defaultdict(lambda: {"dense": Scores(), "bm25": Scores(), "hybrid": Scores()})

    for question in questions:
        query_vector = embedder.embed_query(question.query)

        t0 = time.perf_counter()
        dense_hits = dense_search(query_vector, vectors, top_k=candidate_k)
        dense_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25_hits = bm25.search(question.query, top_k=candidate_k)
        bm25_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits], top_k=candidate_k, weights=list(weights)
        )
        hybrid_ms = dense_ms + bm25_ms + (time.perf_counter() - t0) * 1000

        results = {
            "dense": ([h.chunk_id for h in dense_hits], dense_ms),
            "bm25": ([h.chunk_id for h in bm25_hits], bm25_ms),
            "hybrid": ([f.chunk_id for f in fused], hybrid_ms),
        }

        for method, (retrieved, ms) in results.items():
            overall[method].add(retrieved, question.relevant_chunk_ids, ms)
            by_kind[question.kind][method].add(retrieved, question.relevant_chunk_ids, ms)

    return {
        "overall": overall,
        "by_kind": dict(by_kind),
        "questions": questions,
        "chunk_count": len(chunks_by_id),
        "index_seconds": index_seconds,
    }


def print_report(results, embedder, chunk_size: int, overlap: int) -> None:
    overall = results["overall"]
    by_kind = results["by_kind"]

    print()
    print("=" * 74)
    print(" RETRIEVAL EVALUATION")
    print("=" * 74)
    print(f"  embedder      {type(embedder).__name__} ({embedder.dimensions}d)")
    print(f"  corpus        {len(DOCUMENTS)} documents → {results['chunk_count']} chunks")
    print(f"  chunk size    {chunk_size} chars, {overlap} overlap")
    print(f"  questions     {len(results['questions'])}")
    print(f"  index time    {results['index_seconds']:.2f}s")

    print()
    print(" OVERALL")
    print(" " + "-" * 72)
    print(f"  {'method':<12} {'recall@5':>10} {'recall@10':>11} {'MRR':>8} {'latency':>11}")
    for method in ("dense", "bm25", "hybrid"):
        s = overall[method]
        print(
            f"  {method:<12} {s.mean('recall_5'):>9.1%} {s.mean('recall_10'):>10.1%} "
            f"{s.mean('mrr'):>8.3f} {s.mean('latency_ms'):>9.2f}ms"
        )

    print()
    print(" BY QUESTION TYPE — where each method wins and loses")
    print(" " + "-" * 72)
    labels = {
        "paraphrase": "paraphrase (no shared words)",
        "identifier": "identifier (exact strings)",
        "mixed": "mixed (both signals)",
    }
    for kind in ("paraphrase", "identifier", "mixed"):
        if kind not in by_kind:
            continue
        n = sum(1 for q in results["questions"] if q.kind == kind)
        print(f"\n  {labels[kind]}  ({n} questions)")
        for method in ("dense", "bm25", "hybrid"):
            s = by_kind[kind][method]
            print(f"    {method:<10} recall@5 {s.mean('recall_5'):>6.1%}"
                  f"   recall@10 {s.mean('recall_10'):>6.1%}"
                  f"   MRR {s.mean('mrr'):>5.3f}")

    # --- verdict -----------------------------------------------------------
    #
    # Reported as a finding rather than a pass/fail. Fusion is not guaranteed
    # to win: it helps when the retrievers are comparably strong and hurts when
    # one dominates, because equal-weight RRF lets the weaker list inject noise
    # into the top ranks. Which of those holds is what this measures.
    print()
    print("=" * 74)
    d5, b5, h5 = (overall[m].mean("recall_5") for m in ("dense", "bm25", "hybrid"))
    best_single, best_name = max((d5, "dense"), (b5, "bm25"))

    if h5 >= best_single:
        print(f" FINDING: fusion helps. recall@5 {h5:.1%} vs {best_single:.1%} "
              f"for {best_name} alone (+{(h5 - best_single) * 100:.1f} pts)")
    else:
        gap = (best_single - h5) * 100
        weaker = "bm25" if best_name == "dense" else "dense"
        weaker_score = b5 if best_name == "dense" else d5
        print(f" FINDING: fusion hurts here. recall@5 {h5:.1%} vs {best_single:.1%} "
              f"for {best_name} alone (-{gap:.1f} pts)")
        print()
        print(f" {best_name} ({best_single:.1%}) far outclasses {weaker} "
              f"({weaker_score:.1%}) on this corpus, and equal-weight RRF lets")
        print(f" the weaker list push noise into the top ranks. Try "
              f"--weight-sweep to see")
        print(f" how far leaning toward {best_name} recovers the loss.")
    print("=" * 74)

    if isinstance(embedder, HashEmbedder):
        print()
        print(" NOTE: HashEmbedder is not semantic — dense numbers here are")
        print(" meaningless. Re-run with --embedder local for a real result.")
    print()


def print_examples(results, limit: int = 3) -> None:
    """Show concrete cases where one method fails and the other rescues it.

    Aggregate numbers convince nobody on their own; a reader wants to see the
    query where BM25 returned nothing and dense saved it.
    """
    print(" ILLUSTRATIVE CASES")
    print(" " + "-" * 72)
    for kind, description in (
        ("paraphrase", "dense should win — no vocabulary overlap"),
        ("identifier", "BM25 should win — exact string, no semantics"),
    ):
        examples = [q for q in results["questions"] if q.kind == kind][:limit]
        print(f"\n  {description}")
        for q in examples:
            print(f'    "{q.query}"')
            print(f"      → {q.note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedder", default="local", choices=["local", "hash", "voyage", "openai"],
        help="local = sentence-transformers (default, semantic, offline)",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--examples", action="store_true")
    parser.add_argument(
        "--sweep", action="store_true",
        help="run the chunk-size experiment across several sizes",
    )
    parser.add_argument(
        "--weight-sweep", action="store_true",
        help="vary the dense:bm25 fusion weighting to find where it helps",
    )
    parser.add_argument(
        "--weights", default="1,1",
        help="fusion weights as dense,bm25 (default 1,1)",
    )
    args = parser.parse_args()

    try:
        weights = tuple(float(w) for w in args.weights.split(","))
        if len(weights) != 2:
            raise ValueError
    except ValueError:
        print("--weights must be two numbers, e.g. 3,1", file=sys.stderr)
        return 1

    try:
        embedder = build_embedder(args.embedder, args.api_key, args.model)
    except ImportError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.sweep:
        print()
        print("=" * 74)
        print(" CHUNK SIZE EXPERIMENT")
        print("=" * 74)
        print(f"  {'chunk size':>11} {'chunks':>8} {'recall@5':>10} {'recall@10':>11} {'MRR':>8}")
        for size in (512, 1000, 1500, 2500):
            r = evaluate(embedder, size, args.overlap)
            s = r["overall"]["hybrid"]
            print(f"  {size:>11} {r['chunk_count']:>8} {s.mean('recall_5'):>9.1%} "
                  f"{s.mean('recall_10'):>10.1%} {s.mean('mrr'):>8.3f}")
        print()
        print("  Smaller chunks localise the answer better but fragment context;")
        print("  larger chunks retrieve more surrounding noise per hit.")
        print()
        return 0

    if args.weight_sweep:
        print()
        print("=" * 74)
        print(" FUSION WEIGHT EXPERIMENT")
        print("=" * 74)
        print("  Equal weighting assumes both retrievers are equally trustworthy.")
        print("  When one dominates, leaning toward it should recover the loss.")
        print()
        baseline = evaluate(embedder, args.chunk_size, args.overlap)["overall"]
        d5 = baseline["dense"].mean("recall_5")
        b5 = baseline["bm25"].mean("recall_5")
        print(f"  {'dense:bm25':>12} {'recall@5':>10} {'recall@10':>11} {'MRR':>8}")
        print(f"  {'dense only':>12} {d5:>9.1%} "
              f"{baseline['dense'].mean('recall_10'):>10.1%} "
              f"{baseline['dense'].mean('mrr'):>8.3f}")
        print(f"  {'bm25 only':>12} {b5:>9.1%} "
              f"{baseline['bm25'].mean('recall_10'):>10.1%} "
              f"{baseline['bm25'].mean('mrr'):>8.3f}")
        for w in ((1, 1), (2, 1), (3, 1), (5, 1), (10, 1)):
            r = evaluate(embedder, args.chunk_size, args.overlap, weights=w)
            s = r["overall"]["hybrid"]
            print(f"  {f'{w[0]}:{w[1]}':>12} {s.mean('recall_5'):>9.1%} "
                  f"{s.mean('recall_10'):>10.1%} {s.mean('mrr'):>8.3f}")
        print()
        return 0

    results = evaluate(embedder, args.chunk_size, args.overlap, weights=weights)
    print_report(results, embedder, args.chunk_size, args.overlap)
    if args.examples:
        print_examples(results)

    # Exit code reflects whether the harness ran, not which method won.
    #
    # An earlier version failed the build when hybrid lost. That is the wrong
    # thing to gate on: it turns a measurement into a target, and the moment a
    # legitimate result disagrees you are tempted to adjust the corpus until it
    # agrees. The regression worth catching is retrieval breaking outright, so
    # that is what is asserted.
    overall = results["overall"]
    if not results["questions"] or overall["hybrid"].mean("recall_5") == 0:
        print("evaluation produced no usable results", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
