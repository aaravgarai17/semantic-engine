"""Answer generation with citations.

The two properties that matter
-------------------------------
**Every claim must be traceable.** An answer without citations is
indistinguishable from a hallucination, and the whole point of retrieval
augmentation is that the user can check the source. Chunks are numbered in the
prompt and the model is required to cite by number; citations are then parsed
back out and resolved to real chunks, discarding any the model invented.

**The system must be able to say "I don't know."** A model given irrelevant
context will still answer, confidently, from its parametric memory. That is the
worst failure mode for a document-grounded system, because it looks exactly
like a correct answer. Two defences: a retrieval score floor that refuses
before calling the model at all, and an explicit instruction that refusing is
the correct response when the context does not contain the answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from app.store import StoredChunk

log = logging.getLogger("engine.generate")

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

REFUSAL = (
    "I could not find anything in the indexed documents that answers this "
    "question."
)

SYSTEM_PROMPT = """You answer questions strictly from the provided excerpts.

RULES:
1. Use only the numbered excerpts below. Do not draw on outside knowledge, \
even if you are confident it is correct.
2. Cite the excerpt supporting each claim with its number in square brackets, \
like [2]. Cite multiple with [1, 3].
3. Every factual sentence must carry a citation.
4. If the excerpts do not contain the answer, say so plainly and stop. Do not \
speculate, and do not assemble an answer from loosely related material. \
Refusing is the correct response when the information is absent — it is not a \
failure.
5. Be concise. Answer the question asked, without preamble."""


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        ...


@dataclass
class Citation:
    number: int
    chunk_id: str
    document_title: str
    heading: str = ""
    page: Optional[int] = None
    excerpt: str = ""

    def label(self) -> str:
        bits = [self.document_title]
        if self.page is not None:
            bits.append(f"p.{self.page}")
        if self.heading:
            bits.append(self.heading)
        return " · ".join(b for b in bits if b)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    chunks_considered: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def is_grounded(self) -> bool:
        """Whether the answer cites anything.

        An uncited, non-refusing answer means the model ignored its
        instructions and answered from memory — worth surfacing rather than
        returning silently.
        """
        return self.refused or bool(self.citations)


def build_context(
    chunks: Sequence[StoredChunk],
    max_chars: int = 12_000,
) -> tuple[str, list[StoredChunk]]:
    """Assemble numbered excerpts within a character budget.

    Chunks arrive ranked, so truncation drops the least relevant first.
    Returns the rendered context and the chunks that actually fit — the caller
    needs the latter to resolve citation numbers, which must correspond to what
    the model was shown rather than to what was retrieved.
    """
    parts: list[str] = []
    included: list[StoredChunk] = []
    used = 0

    for number, chunk in enumerate(chunks, start=1):
        header = f"[{number}] {chunk.citation()}"
        block = f"{header}\n{chunk.text}"

        if used + len(block) > max_chars and included:
            break

        parts.append(block)
        included.append(chunk)
        used += len(block)

    return "\n\n---\n\n".join(parts), included


def parse_citations(text: str, chunks: Sequence[StoredChunk]) -> list[Citation]:
    """Extract cited numbers and resolve them to chunks.

    Numbers outside the range shown to the model are dropped: a citation to
    excerpt [9] when only five were provided is a fabrication, and silently
    rendering it would produce a link to nothing.
    """
    seen: dict[int, Citation] = {}

    for match in CITATION.finditer(text):
        for raw in match.group(1).split(","):
            try:
                number = int(raw.strip())
            except ValueError:
                continue

            if not (1 <= number <= len(chunks)):
                log.warning("dropping fabricated citation [%d]", number)
                continue
            if number in seen:
                continue

            chunk = chunks[number - 1]
            seen[number] = Citation(
                number=number,
                chunk_id=chunk.chunk_id,
                document_title=chunk.document_title or chunk.document_id,
                heading=chunk.heading,
                page=chunk.page,
                excerpt=chunk.text[:300],
            )

    return [seen[n] for n in sorted(seen)]


def answer_question(
    llm: LLM,
    question: str,
    chunks: Sequence[StoredChunk],
    scores: Optional[Sequence[float]] = None,
    min_score: float = 0.0,
    max_context_chars: int = 12_000,
    max_tokens: int = 1024,
) -> Answer:
    """Generate a cited answer, or refuse.

    `min_score` refuses *before* spending a request when the best retrieval hit
    is too weak. Cheaper than calling the model and hoping it declines, and more
    reliable — a model shown five irrelevant excerpts will often find a way to
    answer from them.
    """
    if not chunks:
        return Answer(text=REFUSAL, refused=True, chunks_considered=0)

    if scores and min_score > 0 and max(scores) < min_score:
        log.info("refusing: best score %.4f below floor %.4f", max(scores), min_score)
        return Answer(text=REFUSAL, refused=True, chunks_considered=len(chunks))

    context, included = build_context(chunks, max_context_chars)

    user_prompt = (
        f"EXCERPTS:\n\n{context}\n\n"
        f"---\n\nQUESTION: {question}\n\n"
        "Answer using only the excerpts above, citing each claim by number."
    )

    text, input_tokens, output_tokens = llm.complete(
        SYSTEM_PROMPT, user_prompt, max_tokens
    )

    citations = parse_citations(text, included)
    refused = _looks_like_refusal(text)

    if not citations and not refused:
        # The model answered without grounding. Surfaced rather than hidden:
        # an uncited answer in a citation-based system is a bug worth seeing.
        log.warning("answer contains no citations: %s", text[:120])

    return Answer(
        text=text.strip(),
        citations=citations,
        refused=refused,
        chunks_considered=len(included),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "could not find",
            "couldn't find",
            "do not contain",
            "don't contain",
            "does not contain",
            "doesn't contain",
            "no information",
            "not mentioned in",
            "unable to answer",
        )
    )
