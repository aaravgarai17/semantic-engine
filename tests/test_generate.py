"""Answer generation: context assembly, citation parsing, refusal."""

import pytest

from app.generate import (
    REFUSAL,
    Answer,
    answer_question,
    build_context,
    parse_citations,
)
from app.store import StoredChunk


def chunk(n: int, text: str = "", title: str = "Doc", page=None, heading="") -> StoredChunk:
    return StoredChunk(
        chunk_id=f"c{n}",
        document_id="doc1",
        collection="default",
        text=text or f"Content of chunk {n}.",
        embed_text=text or f"Content of chunk {n}.",
        index=n,
        heading=heading,
        page=page,
        document_title=title,
    )


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append((system, user))
        return self.response, 500, 80


# ------------------------------------------------------------- context


def test_context_numbers_excerpts_from_one():
    context, included = build_context([chunk(1), chunk(2)])
    assert "[1]" in context and "[2]" in context
    assert len(included) == 2


def test_context_includes_citation_metadata():
    context, _ = build_context([chunk(1, title="Manual", page=7, heading="Setup")])
    assert "Manual" in context and "p.7" in context and "Setup" in context


def test_context_truncates_at_the_budget():
    chunks = [chunk(i, text="x" * 500) for i in range(20)]
    context, included = build_context(chunks, max_chars=2000)

    assert len(included) < 20
    assert len(context) <= 2600        # allows for headers and separators


def test_truncation_drops_the_least_relevant_first():
    """Chunks arrive ranked, so the tail is what gets cut."""
    chunks = [chunk(i, text="y" * 400) for i in range(10)]
    _, included = build_context(chunks, max_chars=1200)

    assert included[0].chunk_id == "c0"
    assert "c9" not in [c.chunk_id for c in included]


def test_at_least_one_chunk_survives_a_tiny_budget():
    """Returning nothing would guarantee a refusal regardless of relevance."""
    _, included = build_context([chunk(1, text="z" * 5000)], max_chars=100)
    assert len(included) == 1


# ------------------------------------------------------------ citations


def test_parses_a_single_citation():
    citations = parse_citations("The limit is 1000 [1].", [chunk(1), chunk(2)])
    assert len(citations) == 1
    assert citations[0].number == 1
    assert citations[0].chunk_id == "c1"


def test_parses_grouped_citations():
    citations = parse_citations("Both apply [1, 3].", [chunk(i) for i in range(1, 5)])
    assert [c.number for c in citations] == [1, 3]


def test_deduplicates_repeated_citations():
    citations = parse_citations("First [1]. Second [1]. Third [1].", [chunk(1)])
    assert len(citations) == 1


def test_citations_are_returned_in_order():
    citations = parse_citations("Later [3] then earlier [1].", [chunk(i) for i in range(1, 5)])
    assert [c.number for c in citations] == [1, 3]


def test_fabricated_citations_are_dropped():
    """A citation to excerpt [9] when five were shown is an invention.

    Rendering it would produce a link to nothing.
    """
    citations = parse_citations("From [9].", [chunk(1), chunk(2)])
    assert citations == []


def test_zero_and_negative_citations_dropped():
    assert parse_citations("See [0].", [chunk(1)]) == []


def test_no_citations_returns_empty():
    assert parse_citations("An answer with no references at all.", [chunk(1)]) == []


def test_citation_label_composition():
    citations = parse_citations("[1]", [chunk(1, title="Guide", page=4, heading="Intro")])
    assert citations[0].label() == "Guide · p.4 · Intro"


# -------------------------------------------------------------- refusal


def test_refuses_with_no_chunks():
    answer = answer_question(StubLLM("unused"), "anything?", [])
    assert answer.refused
    assert answer.text == REFUSAL


def test_refuses_before_calling_the_model_when_scores_are_too_low():
    """Cheaper and more reliable than hoping the model declines.

    A model shown five irrelevant excerpts will usually find a way to answer
    from them.
    """
    llm = StubLLM("should never be called")
    answer = answer_question(
        llm, "unrelated question", [chunk(1)], scores=[0.05], min_score=0.5
    )

    assert answer.refused
    assert llm.calls == []


def test_proceeds_when_scores_clear_the_floor():
    llm = StubLLM("The answer is 42 [1].")
    answer = answer_question(
        llm, "question", [chunk(1)], scores=[0.9], min_score=0.5
    )
    assert not answer.refused
    assert len(llm.calls) == 1


def test_model_refusal_is_detected():
    llm = StubLLM("The excerpts do not contain information about that topic.")
    answer = answer_question(llm, "unrelated", [chunk(1)])

    assert answer.refused
    assert answer.is_grounded          # refusing counts as grounded


def test_uncited_answer_is_flagged_as_ungrounded():
    """An answer with no citations means the model used outside knowledge."""
    llm = StubLLM("The capital of France is Paris.")
    answer = answer_question(llm, "capital of France?", [chunk(1)])

    assert not answer.refused
    assert not answer.is_grounded


# -------------------------------------------------------------- end to end


def test_full_answer_with_citations():
    llm = StubLLM("The rate limit is 1000 per minute [1], returning 429 [2].")
    chunks = [
        chunk(1, text="The allowance is 1000 requests per minute.", title="API"),
        chunk(2, text="Exceeding it returns HTTP 429.", title="API"),
    ]
    answer = answer_question(llm, "what is the rate limit?", chunks)

    assert not answer.refused
    assert answer.is_grounded
    assert len(answer.citations) == 2
    assert answer.input_tokens == 500
    assert answer.chunks_considered == 2


def test_prompt_forbids_outside_knowledge():
    llm = StubLLM("[1]")
    answer_question(llm, "q", [chunk(1)])

    system, user = llm.calls[0]
    assert "strictly from the provided excerpts" in system.lower()
    assert "outside knowledge" in system.lower()
    assert "refusing is the correct response" in system.lower()
    assert "q" in user
