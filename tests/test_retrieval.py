"""BM25, dense search, and Reciprocal Rank Fusion.

The fusion tests encode the properties RRF is chosen *for* — agreement across
retrievers beating a single strong signal, and rank position mattering rather
than raw scores.
"""

import math

import pytest

from app.retrieval import (
    BM25Index,
    ScoredChunk,
    cosine_similarity,
    dense_search,
    hybrid_search,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank_fusion,
    tokenize,
)


# ------------------------------------------------------------------ tokenizer


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_removes_stopwords():
    assert "the" not in tokenize("the quick brown fox")


def test_tokenize_preserves_identifiers():
    """Identifiers are precisely what BM25 exists to catch.

    Splitting ERR_4032 into 'err' and '4032' would destroy the advantage
    keyword search has over embeddings.
    """
    assert "err_4032" in tokenize("The ERR_4032 code means failure")
    assert "oauth2" in tokenize("Configure oauth2 settings")


def test_tokenize_drops_punctuation():
    assert tokenize("hello, world! (test)") == ["hello", "world", "test"]


# ----------------------------------------------------------------------- BM25


@pytest.fixture
def index():
    idx = BM25Index()
    idx.add_many([
        ("c1", "The password reset flow sends an email to the user."),
        ("c2", "Error ERR_4032 indicates the database connection timed out."),
        ("c3", "Credential recovery requires identity verification first."),
        ("c4", "Database connections are pooled for performance reasons."),
        ("c5", "The user profile page displays account settings."),
    ])
    return idx


def test_exact_identifier_match(index):
    """BM25's core strength: finding a rare literal string."""
    results = index.search("ERR_4032")
    assert results[0].chunk_id == "c2"


def test_ranks_by_relevance(index):
    results = index.search("database connection")
    assert {r.chunk_id for r in results[:2]} == {"c2", "c4"}


def test_no_match_returns_empty(index):
    assert index.search("quantum entanglement photons") == []


def test_stopword_only_query_returns_empty(index):
    assert index.search("the and of") == []


def test_empty_index_returns_empty():
    assert BM25Index().search("anything") == []


def test_results_are_ranked_from_one(index):
    results = index.search("database")
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_top_k_limits_results(index):
    assert len(index.search("the user database connection", top_k=2)) <= 2


def test_rare_terms_score_higher_than_common_ones(index):
    """IDF at work: a term in one document is more informative than one in four."""
    assert index.idf("err_4032") > index.idf("database")


def test_idf_is_never_negative():
    """A term in most documents must not actively push matches down.

    Without the +1 inside the log, a term appearing in more than half the
    corpus scores negative and penalises documents containing it.
    """
    idx = BM25Index()
    idx.add_many([(f"c{i}", "common word here") for i in range(10)])
    assert idx.idf("common") >= 0


def test_term_frequency_saturates(index):
    """The tenth occurrence adds far less than the second.

    Without saturation a keyword-stuffed document would dominate.
    """
    idx = BM25Index()
    idx.add_many([
        ("normal", "database"),
        ("stuffed", " ".join(["database"] * 50)),
    ])
    results = {r.chunk_id: r.score for r in idx.search("database")}
    assert results["stuffed"] < results["normal"] * 3


def test_long_documents_are_penalised():
    """Length normalisation: a long document shouldn't win by size alone."""
    idx = BM25Index()
    idx.add_many([
        ("short", "python testing"),
        ("long", "python testing " + "unrelated filler text " * 60),
    ])
    results = {r.chunk_id: r.score for r in idx.search("python testing")}
    assert results["short"] > results["long"]


# ---------------------------------------------------------------------- dense


def test_cosine_identical_vectors():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """Direction is what matters; a doubled vector is the same direction."""
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_dense_search_ranks_by_similarity():
    vectors = {
        "near": [1.0, 0.0, 0.0],
        "mid": [0.7, 0.7, 0.0],
        "far": [0.0, 0.0, 1.0],
    }
    results = dense_search([1.0, 0.0, 0.0], vectors, top_k=3)
    assert [r.chunk_id for r in results] == ["near", "mid", "far"]


def test_dense_search_respects_top_k():
    vectors = {f"c{i}": [float(i), 1.0] for i in range(10)}
    assert len(dense_search([1.0, 1.0], vectors, top_k=3)) == 3


def test_dense_search_on_empty_store():
    assert dense_search([1.0], {}, top_k=5) == []


# ------------------------------------------------------------------------ RRF


def make_list(ids, source):
    return [
        ScoredChunk(chunk_id=cid, score=1.0, rank=i + 1, source=source)
        for i, cid in enumerate(ids)
    ]


def test_rrf_formula():
    """Score is 1/(k + rank), summed across lists."""
    fused = reciprocal_rank_fusion([make_list(["a"], "dense")], k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_agreement_across_retrievers_wins():
    """The central property of RRF.

    'b' is second in both lists. 'a' is first in one and absent from the other.
    Cross-method agreement should beat a single first place — that agreement is
    the signal fusion exists to exploit.
    """
    dense = make_list(["a", "b", "c"], "dense")
    bm25 = make_list(["d", "b", "e"], "bm25")

    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    assert fused[0].chunk_id == "b"


def test_k_damps_the_advantage_of_rank_one():
    """With small k, first place is nearly unbeatable; with k=60 it isn't.

    This is why the constant exists. `b` sits at rank 4 in both lists, `a` is
    first in one and absent from the other:

        k=1   a = 1/2   = 0.500     b = 2 × 1/5  = 0.400   → a wins
        k=60  a = 1/61  = 0.016     b = 2 × 1/64 = 0.031   → b wins

    A small k gives a steep 1/2, 1/3, 1/4 curve where one top placement beats
    any amount of agreement. k=60 flattens it so consensus decides.
    """
    dense = make_list(["a", "x1", "x2", "b"], "dense")
    bm25 = make_list(["c", "y1", "y2", "b"], "bm25")

    with_small_k = reciprocal_rank_fusion([dense, bm25], k=1)
    with_default = reciprocal_rank_fusion([dense, bm25], k=60)

    assert with_small_k[0].chunk_id == "a"     # rank-1 dominates
    assert with_default[0].chunk_id == "b"     # agreement dominates


def test_scores_are_ignored_only_ranks_matter():
    """A huge raw score confers no advantage — position is all that counts."""
    a = [ScoredChunk("a", score=999.0, rank=2, source="dense"),
         ScoredChunk("b", score=0.001, rank=1, source="dense")]
    # Passed in list order, so 'a' is treated as position 1 despite its rank
    # field — fusion uses enumeration order, making it work on any sequence.
    fused = reciprocal_rank_fusion([a], k=60)
    assert fused[0].chunk_id == "a"


def test_records_which_retrievers_found_each_result():
    dense = make_list(["a", "b"], "dense")
    bm25 = make_list(["b", "c"], "bm25")
    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    by_id = {f.chunk_id: f for f in fused}
    assert by_id["b"].found_by_both
    assert by_id["a"].dense_rank == 1 and by_id["a"].bm25_rank is None
    assert by_id["c"].bm25_rank == 2 and by_id["c"].dense_rank is None


def test_document_found_by_only_one_retriever_still_ranks():
    """Covering a blind spot is the point — a unique find must not be discarded."""
    dense = make_list(["only_dense"], "dense")
    bm25 = make_list(["only_bm25"], "bm25")
    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    assert {f.chunk_id for f in fused} == {"only_dense", "only_bm25"}


def test_weights_can_favour_one_retriever():
    dense = make_list(["a"], "dense")
    bm25 = make_list(["b"], "bm25")

    assert reciprocal_rank_fusion([dense, bm25], weights=[3.0, 1.0])[0].chunk_id == "a"
    assert reciprocal_rank_fusion([dense, bm25], weights=[1.0, 3.0])[0].chunk_id == "b"


def test_mismatched_weights_raise():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([make_list(["a"], "dense")], weights=[1.0, 1.0])


def test_fusion_is_deterministic():
    """The eval harness must produce identical numbers on every run."""
    dense = make_list(["a", "b", "c"], "dense")
    bm25 = make_list(["c", "b", "a"], "bm25")

    first = [f.chunk_id for f in reciprocal_rank_fusion([dense, bm25])]
    second = [f.chunk_id for f in reciprocal_rank_fusion([dense, bm25])]
    assert first == second


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_top_k_applies_after_fusion():
    dense = make_list([f"c{i}" for i in range(20)], "dense")
    assert len(reciprocal_rank_fusion([dense], top_k=5)) == 5


# ---------------------------------------------------------------- hybrid


def test_hybrid_finds_what_each_method_alone_would_miss():
    """The whole thesis, on a corpus built to expose both failure modes."""
    corpus = {
        "sem": "Credential recovery requires verifying your identity.",
        "lex": "Fault code ERR_4032 signals a timeout.",
        "noise": "The cafeteria serves lunch from noon.",
    }
    bm25 = BM25Index()
    bm25.add_many(corpus.items())

    # Hand-built vectors: 'sem' is semantically near a password-reset query,
    # the others are not.
    vectors = {"sem": [1.0, 0.0], "lex": [0.0, 1.0], "noise": [0.5, 0.5]}

    # A paraphrase query — no shared words with 'sem', so BM25 alone fails.
    fused = hybrid_search("how do I reset my password", [1.0, 0.0], bm25, vectors)
    assert fused[0].chunk_id == "sem"

    # An identifier query — embeddings smear it, so dense alone fails.
    fused = hybrid_search("ERR_4032", [0.5, 0.5], bm25, vectors)
    assert fused[0].chunk_id == "lex"


def test_hybrid_retrieves_deeper_than_it_returns():
    """Fusion can only reorder what it is given.

    A chunk ranked 20th by dense and 3rd by BM25 needs both candidate lists to
    run deep enough to contain it.
    """
    bm25 = BM25Index()
    bm25.add_many([(f"c{i}", f"document number {i} about testing") for i in range(40)])
    vectors = {f"c{i}": [float(i % 7), 1.0] for i in range(40)}

    fused = hybrid_search("testing", [1.0, 1.0], bm25, vectors, top_k=5, candidate_k=30)
    assert len(fused) == 5


# ------------------------------------------------------------------- metrics


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "x", "y"], ["a", "c"], k=3) == 0.5
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0


def test_recall_with_no_relevant_documents():
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_mrr_rewards_ranking_the_answer_first():
    assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert mean_reciprocal_rank(["b", "a"], ["a"]) == 0.5
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0
