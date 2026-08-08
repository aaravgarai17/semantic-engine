"""Splitting documents into retrievable chunks.

Why chunk at all
----------------
Embeddings are fixed-length regardless of input size, so embedding a whole
document compresses everything into one point and loses the ability to say
*which part* answered the question. Citations need passage-level granularity,
and the generator needs a context window's worth of the most relevant text
rather than an entire manual.

Why fixed-size chunking is bad
------------------------------
The naive approach — every 512 tokens, hard stop — cuts sentences in half,
splits tables from their headers, and separates a heading from the section it
introduces. Retrieval quality drops for a specific and avoidable reason: a
chunk severed mid-sentence embeds as something close to gibberish, so it
matches nothing, and the answer it contained becomes unreachable.

The strategy here
-----------------
Split on structure first, fall back to size only where a section is genuinely
too large:

  1. Break on **headings** — markdown `#`, underlined titles, numbered sections.
     A heading marks a topic change, which is exactly a chunk boundary.
  2. Within an oversized section, break on **paragraphs** (blank lines).
  3. Within an oversized paragraph, break on **sentences**.
  4. Only if a single sentence exceeds the budget do we cut mid-sentence.

Each chunk carries its heading trail, so a chunk from "Installation > Linux"
knows its context even though the words "Installation" and "Linux" may appear
nowhere in its body. That both improves retrieval (the heading is prepended to
the embedded text) and makes citations legible.

Overlap
-------
Adjacent chunks share a configurable tail. Without it, a fact stated across a
boundary — a definition at the end of one chunk, its consequence at the start
of the next — is retrievable from neither. Overlap costs storage and some
duplicate retrieval; a few hundred characters is the usual point where the
recall gain stops justifying the cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

# `# Heading` through `###### Heading`
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
# `Heading` followed by `=====` or `-----`
SETEXT_UNDERLINE = re.compile(r"^(=|-){3,}\s*$")
# `1.2.3 Section Title`
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


@dataclass
class Chunk:
    """One retrievable passage."""

    text: str
    index: int
    heading_path: list[str] = field(default_factory=list)
    page: Optional[int] = None
    char_start: int = 0
    char_end: int = 0

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def embed_text(self) -> str:
        """Text actually embedded — the heading trail prepended to the body.

        A chunk deep inside "Troubleshooting > Network > DNS" often never
        repeats those words, so without this it is unreachable by a query
        about DNS troubleshooting. Prepending costs a few tokens and
        substantially improves recall on sectioned documents.
        """
        if not self.heading_path:
            return self.text
        return f"{self.heading}\n\n{self.text}"

    @property
    def citation_label(self) -> str:
        if self.page is not None:
            return f"p.{self.page}" + (f" · {self.heading}" if self.heading else "")
        return self.heading or f"chunk {self.index}"


@dataclass
class Section:
    """A run of text under one heading trail."""

    text: str
    heading_path: list[str]
    char_start: int
    page: Optional[int] = None


def split_into_sections(text: str) -> list[Section]:
    """Break a document at heading boundaries, tracking nesting depth.

    The heading *path* is maintained as a stack so a level-3 heading inherits
    the level-1 and level-2 headings above it, giving each chunk its full
    context rather than just its immediate parent.
    """
    lines = text.splitlines()
    sections: list[Section] = []

    stack: list[tuple[int, str]] = []          # (level, title)
    buffer: list[str] = []
    buffer_start = 0
    offset = 0

    def flush(start: int) -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                Section(
                    text=body,
                    heading_path=[title for _, title in stack],
                    char_start=start,
                )
            )
        buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        line_length = len(line) + 1

        heading: Optional[tuple[int, str]] = None

        md = MD_HEADING.match(line.strip())
        if md:
            heading = (len(md.group(1)), md.group(2).strip())

        # Setext: a line of text underlined by === or ---
        elif (
            i + 1 < len(lines)
            and line.strip()
            and SETEXT_UNDERLINE.match(lines[i + 1].strip())
        ):
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            heading = (level, line.strip())
            i += 1
            line_length += len(lines[i]) + 1

        else:
            numbered = NUMBERED_HEADING.match(line.strip())
            if numbered:
                depth = numbered.group(1).count(".") + 1
                heading = (depth, numbered.group(2).strip())

        if heading is not None:
            flush(buffer_start)
            level, title = heading
            # Pop deeper-or-equal headings; this heading replaces them.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buffer_start = offset + line_length
        else:
            if not buffer:
                buffer_start = offset
            buffer.append(line)

        offset += line_length
        i += 1

    flush(buffer_start)
    return sections


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Approximate sentence splitting.

    Deliberately not a full NLP dependency: this only runs as a fallback inside
    an oversized paragraph, where the cost of an occasional bad split is far
    lower than the cost of shipping a parser model.
    """
    parts = SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack(
    pieces: list[str],
    max_chars: int,
    joiner: str = "\n\n",
) -> list[str]:
    """Greedily combine pieces into groups under `max_chars`."""
    groups: list[str] = []
    current: list[str] = []
    length = 0

    for piece in pieces:
        piece_length = len(piece) + len(joiner)
        if current and length + piece_length > max_chars:
            groups.append(joiner.join(current))
            current, length = [], 0
        current.append(piece)
        length += piece_length

    if current:
        groups.append(joiner.join(current))
    return groups


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort: cut at a word boundary near the limit."""
    out: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars            # no space at all: cut mid-word
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        out.append(remaining)
    return out


def _apply_overlap(texts: list[str], overlap_chars: int) -> list[str]:
    """Prepend a tail of each chunk to the next.

    Cut at a word boundary so the overlap doesn't begin mid-word, which would
    embed as noise.
    """
    if overlap_chars <= 0 or len(texts) < 2:
        return texts

    out = [texts[0]]
    for previous, current in zip(texts, texts[1:]):
        tail = previous[-overlap_chars:]
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        out.append(f"{tail} {current}".strip() if tail else current)
    return out


def chunk_document(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> list[Chunk]:
    """Split a document into overlapping, structure-aware chunks.

    Sizes are in characters rather than tokens: it avoids a tokenizer
    dependency in the hot path, and roughly four characters per token makes the
    conversion easy to reason about. 1500 characters is about 375 tokens.
    """
    if not text or not text.strip():
        return []

    chunks: list[Chunk] = []
    index = 0

    for section in split_into_sections(text):
        pieces: list[str]

        if len(section.text) <= max_chars:
            pieces = [section.text]
        else:
            paragraphs = split_paragraphs(section.text)
            pieces = []

            for group in _pack(paragraphs, max_chars):
                if len(group) <= max_chars:
                    pieces.append(group)
                    continue
                # A single paragraph over budget: fall back to sentences.
                sentences = split_sentences(group)
                for sentence_group in _pack(sentences, max_chars, joiner=" "):
                    if len(sentence_group) <= max_chars:
                        pieces.append(sentence_group)
                    else:
                        pieces.extend(_hard_split(sentence_group, max_chars))

        pieces = _apply_overlap(pieces, overlap_chars)

        for piece in pieces:
            stripped = piece.strip()
            if not stripped:
                continue

            # Drop fragments too small to carry meaning — unless they're all
            # this section has, in which case a short chunk beats losing it.
            if len(stripped) < min_chars and len(pieces) > 1:
                continue

            chunks.append(
                Chunk(
                    text=stripped,
                    index=index,
                    heading_path=list(section.heading_path),
                    char_start=section.char_start,
                    char_end=section.char_start + len(stripped),
                )
            )
            index += 1

    return chunks
