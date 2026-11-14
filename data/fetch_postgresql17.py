"""Fetch and prepare a small corpus from the official PostgreSQL 17 docs."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
BASE_URL = "https://www.postgresql.org/docs/17/"
SOURCE_ID = "postgresql-17-performance"
MIN_UNIT_CHARS = 1_500
TARGET_UNIT_CHARS = 3_000
MAX_UNIT_CHARS = 4_000

SECTIONS = [
    ("11", "indexes-intro.html", "11.1 Introduction"),
    ("11", "indexes-types.html", "11.2 Index Types"),
    ("11", "indexes-multicolumn.html", "11.3 Multicolumn Indexes"),
    ("11", "indexes-ordering.html", "11.4 Indexes and ORDER BY"),
    ("11", "indexes-bitmap-scans.html", "11.5 Combining Multiple Indexes"),
    ("11", "indexes-unique.html", "11.6 Unique Indexes"),
    ("11", "indexes-expressional.html", "11.7 Indexes on Expressions"),
    ("11", "indexes-partial.html", "11.8 Partial Indexes"),
    ("11", "indexes-index-only-scans.html", "11.9 Index-Only Scans and Covering Indexes"),
    ("11", "indexes-opclass.html", "11.10 Operator Classes and Operator Families"),
    ("11", "indexes-collations.html", "11.11 Indexes and Collations"),
    ("11", "indexes-examine.html", "11.12 Examining Index Usage"),
    ("14", "using-explain.html", "14.1 Using EXPLAIN"),
    ("14", "planner-stats.html", "14.2 Statistics Used by the Planner"),
    ("14", "explicit-joins.html", "14.3 Controlling the Planner with Explicit JOIN Clauses"),
    ("14", "populate.html", "14.4 Populating a Database"),
    ("14", "non-durability.html", "14.5 Non-Durable Settings"),
    ("15", "how-parallel-query-works.html", "15.1 How Parallel Query Works"),
    ("15", "when-can-parallel-query-be-used.html", "15.2 When Can Parallel Query Be Used?"),
    ("15", "parallel-plans.html", "15.3 Parallel Plans"),
    ("15", "parallel-safety.html", "15.4 Parallel Safety"),
]

LICENSE_NOTICE = (
    "PostgreSQL Database Management System (also known as Postgres, formerly known as "
    "Postgres95)\n\n"
    "Portions Copyright © 1996-2026, PostgreSQL Global Development Group\n\n"
    "Portions Copyright © 1994, The Regents of the University of California\n\n"
    "Permission to use, copy, modify, and distribute this software and its documentation "
    "for any purpose, without fee, and without a written agreement is hereby granted, "
    "provided that the above copyright notice and this paragraph and the following two "
    "paragraphs appear in all copies.\n\n"
    "IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, "
    "INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, "
    "ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY "
    "OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\n"
    "THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT "
    "NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A "
    "PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS-IS” BASIS, AND THE "
    "UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, "
    "ENHANCEMENTS, OR MODIFICATIONS."
)


class SectionTextParser(HTMLParser):
    """Extract prose and code blocks from the main section, excluding page navigation."""

    BLOCK_TAGS = {
        "blockquote",
        "dd",
        "dt",
        "h2",
        "h3",
        "h4",
        "li",
        "p",
        "pre",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active = False
        self.section_depth = 0
        self.toc_depth = 0
        self.skip_id_link = False
        self.output: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "div":
            if self.active:
                self.section_depth += 1
            elif "sect1" in classes:
                self.active = True
                self.section_depth = 1
            if self.active and self.toc_depth:
                self.toc_depth += 1
            elif self.active and "toc" in classes:
                self.toc_depth = 1
        if self.active and not self.toc_depth and not self.skip_id_link:
            if tag in self.BLOCK_TAGS or tag == "br":
                self.output.append("\n")
            if tag == "a" and "id_link" in classes:
                self.skip_id_link = True

    def handle_endtag(self, tag: str) -> None:
        if self.active and tag == "a" and self.skip_id_link:
            self.skip_id_link = False
        if self.active and tag == "div":
            if self.toc_depth:
                self.toc_depth -= 1
            self.section_depth -= 1
            if self.section_depth == 0:
                self.active = False
        if self.active and not self.toc_depth and not self.skip_id_link:
            if tag in self.BLOCK_TAGS or tag == "br":
                self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if self.active and not self.toc_depth and not self.skip_id_link:
            self.output.append(data)


def fetch_paragraphs(filename: str) -> list[str]:
    request = Request(
        BASE_URL + filename,
        headers={"User-Agent": "DataIngestionEngineeringCoursework/1.0"},
    )
    try:
        with urlopen(request, timeout=45) as response:
            html = response.read().decode("utf-8")
    except (OSError, URLError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Unable to fetch official source {BASE_URL + filename}: {error}") from error

    parser = SectionTextParser()
    parser.feed(html)
    text = unicodedata.normalize("NFC", "".join(parser.output))
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in text.splitlines()
    ]
    return [paragraph for paragraph in paragraphs if paragraph]


def split_long_text(text: str) -> list[str]:
    pieces: list[str] = []
    remainder = text
    while len(remainder) > MAX_UNIT_CHARS:
        split_at = remainder.rfind(" ", 0, MAX_UNIT_CHARS + 1)
        if split_at < MIN_UNIT_CHARS:
            split_at = MAX_UNIT_CHARS
        pieces.append(remainder[:split_at].strip())
        remainder = remainder[split_at:].strip()
    if remainder:
        pieces.append(remainder)
    return pieces


def build_units(pages: list[dict[str, object]]) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    pending: list[tuple[str, dict[str, object]]] = []

    def finish_pending() -> None:
        if not pending:
            return
        pending_length = sum(len(text) for text, _ in pending) + 2 * (len(pending) - 1)
        pending_chapter = str(pending[0][1]["chapter"])
        if (
            pending_length < MIN_UNIT_CHARS
            and units
            and units[-1]["chapter"] == pending_chapter
        ):
            previous = units[-1]
            tail_text = "\n\n".join(text for text, _ in pending)
            previous["text"] = str(previous["text"]) + "\n\n" + tail_text
            previous["char_count"] = len(str(previous["text"]))
            previous["section_ids"] = list(
                dict.fromkeys(
                    list(previous["section_ids"])
                    + [str(meta["section_id"]) for _, meta in pending]
                )
            )
            previous["source_urls"] = list(
                dict.fromkeys(
                    list(previous["source_urls"])
                    + [str(meta["url"]) for _, meta in pending]
                )
            )
            pending.clear()
        else:
            flush()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        text = "\n\n".join(paragraph for paragraph, _ in pending)
        sections = list(dict.fromkeys(str(meta["section_id"]) for _, meta in pending))
        urls = list(dict.fromkeys(str(meta["url"]) for _, meta in pending))
        chapters = list(dict.fromkeys(str(meta["chapter"]) for _, meta in pending))
        units.append(
            {
                "unit_id": f"pg17-{len(units) + 1:03d}",
                "source_id": SOURCE_ID,
                "title": "PostgreSQL 17: Indexes, Performance, and Parallel Query",
                "chapter": chapters[0] if len(chapters) == 1 else chapters,
                "section_ids": sections,
                "source_urls": urls,
                "version": "PostgreSQL 17",
                "text": text,
                "char_count": len(text),
            }
        )
        pending = []

    active_chapter: str | None = None
    for page in pages:
        chapter = str(page["chapter"])
        if active_chapter is not None and chapter != active_chapter:
            finish_pending()
        active_chapter = chapter
        metadata = {
            "chapter": chapter,
            "section_id": page["section_id"],
            "url": page["url"],
        }
        for paragraph in page["paragraphs"]:
            for piece in split_long_text(str(paragraph)):
                current_length = sum(len(existing) + 2 for existing, _ in pending)
                if current_length >= MIN_UNIT_CHARS and (
                    current_length + len(piece) + 2 > MAX_UNIT_CHARS
                    or current_length >= TARGET_UNIT_CHARS
                ):
                    flush()
                    current_length = 0
                pending.append((piece, metadata))
                current_length += len(piece) + 2
                if current_length >= TARGET_UNIT_CHARS:
                    flush()

    finish_pending()

    for unit in units:
        if int(unit["char_count"]) < MIN_UNIT_CHARS:
            raise RuntimeError(f"{unit['unit_id']} is shorter than {MIN_UNIT_CHARS} characters")
    return units


def main() -> None:
    pages: list[dict[str, object]] = []
    for chapter, filename, title in SECTIONS:
        paragraphs = fetch_paragraphs(filename)
        if not paragraphs:
            raise RuntimeError(f"No section text extracted from {BASE_URL + filename}")
        section_id = filename.removesuffix(".html")
        pages.append(
            {
                "chapter": chapter,
                "section_id": section_id,
                "title": title,
                "url": BASE_URL + filename,
                "char_count": len("\n\n".join(paragraphs)),
                "paragraphs": paragraphs,
            }
        )

    units = build_units(pages)
    manifest = {
        "source_id": SOURCE_ID,
        "title": "PostgreSQL 17 Documentation: Indexes, Performance Tips, and Parallel Query",
        "source_path_or_url": "https://www.postgresql.org/docs/17/index.html",
        "license_or_permission": (
            "PostgreSQL documentation legal notice grants permission to use, copy, modify, "
            "and distribute the documentation provided the notice is retained."
        ),
        "license_source_url": "https://www.postgresql.org/docs/17/legalnotice.html",
        "license_notice": LICENSE_NOTICE,
        "accessed_at": date.today().isoformat(),
        "snapshot": (
            "PostgreSQL 17.12 Documentation; official section URLs use the /17/ series path."
        ),
        "selection_rule": (
            "Use the 21 content sections listed below from chapters 11, 14, and 15; "
            "navigation, header, and footer text are excluded. HTML is normalized to NFC; "
            "unit char_count uses Python len() on normalized text."
        ),
        "minimum_unit_characters": MIN_UNIT_CHARS,
        "section_count": len(pages),
        "unit_count": len(units),
        "sections": [
            {
                "chapter": page["chapter"],
                "section_id": page["section_id"],
                "title": page["title"],
                "source_path_or_url": page["url"],
                "normalized_character_count": page["char_count"],
            }
            for page in pages
        ],
        "units": [
            {
                "unit_id": unit["unit_id"],
                "section_ids": unit["section_ids"],
                "normalized_character_count": unit["char_count"],
            }
            for unit in units
        ],
    }

    (ROOT / "postgresql17_units.jsonl").write_text(
        "".join(json.dumps(unit, ensure_ascii=False) + "\n" for unit in units),
        encoding="utf-8",
    )
    (ROOT / "postgresql17_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Fetched {len(pages)} sections and wrote {len(units)} units.")
    print(f"Unit lengths: {min(int(u['char_count']) for u in units)}–"
          f"{max(int(u['char_count']) for u in units)} characters.")


if __name__ == "__main__":
    main()
