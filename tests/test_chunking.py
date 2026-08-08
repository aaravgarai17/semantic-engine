"""Structure-aware chunking."""

import pytest

from app.chunking import (
    Chunk,
    chunk_document,
    split_into_sections,
    split_paragraphs,
    split_sentences,
)

DOC = """# Installation

Install the package with pip. It requires Python 3.10 or later.

## Linux

On Debian systems you also need libpq-dev installed first.

## macOS

Use Homebrew to install the dependencies.

# Configuration

Settings live in a YAML file at the project root.
"""


# ----------------------------------------------------------------- sections


def test_splits_on_markdown_headings():
    sections = split_into_sections(DOC)
    assert len(sections) == 4


def test_heading_path_records_nesting():
    """A level-2 heading inherits the level-1 above it.

    Without the trail, a chunk under 'Installation > Linux' loses the fact that
    it is about installation at all.
    """
    sections = split_into_sections(DOC)
    paths = [s.heading_path for s in sections]

    assert paths[0] == ["Installation"]
    assert paths[1] == ["Installation", "Linux"]
    assert paths[2] == ["Installation", "macOS"]
    assert paths[3] == ["Configuration"]


def test_sibling_heading_replaces_rather_than_nests():
    """'macOS' must not end up nested under 'Linux'."""
    sections = split_into_sections(DOC)
    assert "Linux" not in sections[2].heading_path


def test_deeper_heading_nests():
    text = "# A\n\ntext a\n\n## B\n\ntext b\n\n### C\n\ntext c\n"
    paths = [s.heading_path for s in split_into_sections(text)]
    assert paths[-1] == ["A", "B", "C"]


def test_setext_headings():
    text = "Title\n=====\n\nBody text here.\n\nSubtitle\n--------\n\nMore body.\n"
    sections = split_into_sections(text)

    assert sections[0].heading_path == ["Title"]
    assert sections[1].heading_path == ["Title", "Subtitle"]


def test_numbered_headings():
    text = "1. Introduction\n\nSome text.\n\n1.1 Background\n\nMore text.\n"
    paths = [s.heading_path for s in split_into_sections(text)]

    assert paths[0] == ["Introduction"]
    assert paths[1] == ["Introduction", "Background"]


def test_document_without_headings():
    sections = split_into_sections("Just some text with no structure at all.")
    assert len(sections) == 1
    assert sections[0].heading_path == []


def test_empty_sections_are_dropped():
    """A heading immediately followed by another contributes no text."""
    sections = split_into_sections("# A\n\n# B\n\nOnly B has body text.\n")
    assert len(sections) == 1
    assert sections[0].heading_path == ["B"]


# -------------------------------------------------------------- paragraphs


def test_split_paragraphs():
    assert split_paragraphs("One.\n\nTwo.\n\n\nThree.") == ["One.", "Two.", "Three."]


def test_split_sentences():
    text = "First sentence. Second one here! Third? Yes."
    assert len(split_sentences(text)) == 4


def test_sentence_split_leaves_abbreviations_alone_enough():
    """Approximate splitting: lowercase after a period isn't a boundary."""
    assert len(split_sentences("See fig. 2 for details.")) == 1


# ------------------------------------------------------------------ chunking


def test_chunks_carry_their_heading_path():
    chunks = chunk_document(DOC, max_chars=1000, overlap_chars=0)
    linux = next(c for c in chunks if "Debian" in c.text)
    assert linux.heading_path == ["Installation", "Linux"]
    assert linux.heading == "Installation > Linux"


def test_embed_text_prepends_the_heading():
    """The heading is embedded with the body.

    A chunk under 'Troubleshooting > Network > DNS' may never repeat those
    words, making it unreachable by a query about DNS without this.
    """
    chunks = chunk_document(DOC, max_chars=1000, overlap_chars=0)
    linux = next(c for c in chunks if "Debian" in c.text)

    assert linux.embed_text.startswith("Installation > Linux")
    assert "Debian" in linux.embed_text


def test_chunks_are_indexed_in_order():
    chunks = chunk_document(DOC, max_chars=1000, overlap_chars=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_document():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_respects_max_chars():
    text = "# Title\n\n" + ("This is a sentence of reasonable length. " * 200)
    chunks = chunk_document(text, max_chars=500, overlap_chars=0)
    assert all(len(c.text) <= 500 for c in chunks)


def test_prefers_paragraph_boundaries_over_hard_cuts():
    """Splitting mid-sentence embeds as near-gibberish and matches nothing."""
    paragraphs = "\n\n".join(f"Paragraph number {i} with some content." for i in range(20))
    chunks = chunk_document(paragraphs, max_chars=200, overlap_chars=0)

    # Every chunk should begin at a paragraph start.
    assert all(c.text.startswith("Paragraph") for c in chunks)


def test_falls_back_to_sentences_inside_a_huge_paragraph():
    huge = " ".join(f"Sentence number {i} appears here." for i in range(100))
    chunks = chunk_document(huge, max_chars=300, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_hard_split_only_when_a_sentence_exceeds_the_budget():
    text = "word " * 500      # one enormous "sentence"
    chunks = chunk_document(text, max_chars=200, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_overlap_carries_context_across_boundaries():
    """A fact split across a boundary must remain retrievable.

    Without overlap, a definition ending one chunk and its consequence
    beginning the next are retrievable from neither.
    """
    paragraphs = "\n\n".join(f"Distinct paragraph {i} content here." for i in range(10))

    without = chunk_document(paragraphs, max_chars=120, overlap_chars=0)
    with_overlap = chunk_document(paragraphs, max_chars=120, overlap_chars=40)

    assert len(with_overlap) == len(without)
    assert len(with_overlap[1].text) > len(without[1].text)


def test_overlap_does_not_start_mid_word():
    """A fragment like 'ragraph' embeds as noise."""
    paragraphs = "\n\n".join(f"Paragraph {i} with content." for i in range(10))
    chunks = chunk_document(paragraphs, max_chars=100, overlap_chars=30)

    for chunk in chunks[1:]:
        assert not chunk.text.startswith(" ")
        first_word = chunk.text.split()[0]
        assert first_word.isalnum() or first_word[0].isupper() or "." in first_word


def test_first_chunk_has_no_overlap():
    chunks = chunk_document(DOC, max_chars=200, overlap_chars=50)
    assert chunks[0].text.startswith("Install")


def test_tiny_fragments_are_dropped():
    text = "# Title\n\n" + "Long enough paragraph to survive filtering. " * 10 + "\n\nx\n"
    chunks = chunk_document(text, max_chars=300, overlap_chars=0, min_chars=50)
    assert all(len(c.text) >= 50 for c in chunks)


def test_a_short_document_is_kept_even_below_min_chars():
    """Dropping the only chunk would lose the document entirely."""
    chunks = chunk_document("Tiny.", max_chars=1000, overlap_chars=0, min_chars=100)
    assert len(chunks) == 1


def test_citation_label_uses_heading():
    chunks = chunk_document(DOC, max_chars=1000, overlap_chars=0)
    linux = next(c for c in chunks if "Debian" in c.text)
    assert "Installation > Linux" in linux.citation_label


def test_citation_label_prefers_page_when_present():
    chunk = Chunk(text="body", index=0, heading_path=["A"], page=7)
    assert chunk.citation_label.startswith("p.7")


def test_no_content_is_lost():
    """Every distinctive token in the source should survive into some chunk."""
    chunks = chunk_document(DOC, max_chars=200, overlap_chars=30)
    combined = " ".join(c.text for c in chunks)

    for word in ("Debian", "Homebrew", "YAML", "pip"):
        assert word in combined
