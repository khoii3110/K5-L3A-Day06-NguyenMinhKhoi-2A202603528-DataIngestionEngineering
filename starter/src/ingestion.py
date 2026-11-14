"""Read and normalize source data while preserving source locations."""
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; empty text remains visible for OCR review."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"empty_extraction": not bool(text)},
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def load_jsonl(path: Path) -> list[DocumentPage]:
    """Load prepared source units from the PostgreSQL documentation corpus."""
    pages = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw in enumerate(file, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            text = normalize_text(record["text"])
            if len(text) < 1_500:
                raise ValueError(
                    f"{path}:{line_number}: source unit {record.get('unit_id')} "
                    f"has only {len(text)} normalized characters"
                )
            urls = record["source_urls"]
            if not urls:
                raise ValueError(f"{path}:{line_number}: source unit has no source URL")
            pages.append(
                DocumentPage(
                    document_id=record["source_id"],
                    source=urls[0],
                    page=record["unit_id"],
                    text=text,
                    metadata={
                        "unit_id": record["unit_id"],
                        "section_ids": record["section_ids"],
                        "chapter": record["chapter"],
                        "version": record["version"],
                        "source_urls": urls,
                    },
                )
            )
    if not pages:
        raise ValueError(f"No source units found in {path}")
    return pages
