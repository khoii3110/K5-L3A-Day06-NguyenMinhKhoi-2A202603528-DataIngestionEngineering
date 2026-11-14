"""Fixed-size and paragraph-aware chunking with stable source offsets."""
import hashlib
import re
from .models import Chunk, DocumentPage


def _chunk_id(strategy: str, page: DocumentPage, start: int, end: int, text: str) -> str:
    value = f"{strategy}\0{page.document_id}\0{page.page}\0{start}\0{end}\0{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_chunk(
    strategy: str,
    page: DocumentPage,
    start: int,
    end: int,
    text: str,
) -> Chunk:
    metadata = dict(page.metadata)
    metadata.update({"char_start": start, "char_end": end})
    return Chunk(
        chunk_id=_chunk_id(strategy, page, start, end, text),
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def _split_span(text: str, start: int, max_size: int) -> list[tuple[int, int, str]]:
    spans = []
    offset = 0
    while offset < len(text):
        end = min(offset + max_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", offset, end + 1)
            if boundary > offset:
                end = boundary
        piece = text[offset:end].strip()
        if piece:
            left_trim = len(text[offset:end]) - len(text[offset:end].lstrip())
            right_trim = len(text[offset:end]) - len(text[offset:end].rstrip())
            spans.append((start + offset + left_trim, start + end - right_trim, piece))
        offset = end
        while offset < len(text) and text[offset].isspace():
            offset += 1
    return spans


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split into fixed character windows and retain each window's source offsets."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        text = page.text[start:end].strip()
        if text:
            chunks.append(_make_chunk("fixed", page, start, end, text))
        if end == len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 1_000) -> list[Chunk]:
    """Group adjacent paragraphs up to max_size and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    paragraphs: list[tuple[int, int, str]] = []
    offset = 0
    for paragraph in page.text.split("\n\n"):
        start = page.text.find(paragraph, offset)
        end = start + len(paragraph)
        if paragraph.strip():
            paragraphs.append((start, end, paragraph.strip()))
        offset = end + 2

    spans: list[tuple[int, int, str]] = []
    pending: list[str] = []
    pending_start = 0
    pending_end = 0
    for start, end, paragraph in paragraphs:
        pieces = _split_span(paragraph, start, max_size) if len(paragraph) > max_size else [(start, end, paragraph)]
        for piece_start, piece_end, piece in pieces:
            pending_length = sum(len(item) for item in pending) + 2 * max(0, len(pending) - 1)
            if pending and pending_length + len(piece) + 2 > max_size:
                spans.append((pending_start, pending_end, "\n\n".join(pending)))
                pending = []
            if not pending:
                pending_start = piece_start
            pending.append(piece)
            pending_end = piece_end
            if sum(len(item) for item in pending) + 2 * (len(pending) - 1) >= max_size:
                spans.append((pending_start, pending_end, "\n\n".join(pending)))
                pending = []

    if pending:
        spans.append((pending_start, pending_end, "\n\n".join(pending)))
    return [_make_chunk("paragraph", page, start, end, text) for start, end, text in spans]
