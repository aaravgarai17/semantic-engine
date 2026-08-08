"""Text extraction from uploaded documents.

Page numbers are tracked where the format has them, because a citation reading
"p.14" is far more useful than one pointing at "chunk 37". PDFs and DOCX carry
that information; plain text does not.

Extractors are imported lazily so the package works with only the formats you
actually installed support for — a deployment handling markdown shouldn't need
a PDF library present to start.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("engine.extract")

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class ExtractionError(Exception):
    pass


@dataclass
class ExtractedPage:
    text: str
    page: Optional[int] = None


@dataclass
class ExtractedDocument:
    text: str
    pages: list[ExtractedPage]
    page_count: int
    format: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def extract(path: str | Path) -> ExtractedDocument:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED:
        raise ExtractionError(
            f"unsupported format {suffix!r}; supported: {', '.join(sorted(SUPPORTED))}"
        )
    if not path.exists():
        raise ExtractionError(f"file not found: {path}")

    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    return _extract_text(path)


def _extract_text(path: Path) -> ExtractedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDocument(
        text=_clean(raw),
        pages=[ExtractedPage(text=raw)],
        page_count=1,
        format=path.suffix.lstrip("."),
    )


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExtractionError("PDF support needs pypdf: pip install pypdf") from exc

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"could not read PDF: {exc}") from exc

    pages: list[ExtractedPage] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # One malformed page shouldn't lose the other 299.
            log.warning("could not extract page %d of %s: %s", number, path.name, exc)
            text = ""
        pages.append(ExtractedPage(text=_clean(text), page=number))

    combined = "\n\n".join(p.text for p in pages if p.text.strip())

    if not combined.strip():
        raise ExtractionError(
            "no text found — the PDF is probably a scan and would need OCR"
        )

    return ExtractedDocument(
        text=combined, pages=pages, page_count=len(pages), format="pdf"
    )


def _extract_docx(path: Path) -> ExtractedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("DOCX support needs python-docx: pip install python-docx") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"could not read DOCX: {exc}") from exc

    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Word's built-in heading styles are converted to markdown so the
        # chunker's structure detection works on them.
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            match = re.search(r"(\d+)", style)
            level = int(match.group(1)) if match else 1
            parts.append(f"{'#' * min(level, 6)} {text}")
        else:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    combined = _clean("\n\n".join(parts))
    return ExtractedDocument(
        text=combined,
        pages=[ExtractedPage(text=combined)],
        page_count=1,
        format="docx",
    )


def _clean(text: str) -> str:
    """Normalise extracted text.

    PDF extraction in particular produces stray form feeds, non-breaking
    spaces, and runs of blank lines. Left alone these confuse the chunker's
    paragraph detection, which splits on blank lines.
    """
    text = text.replace("\x0c", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()
