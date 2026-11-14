"""Evidence retrieval metrics and machine-readable top-k traces."""
import re
import unicodedata
from typing import Any


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Count a hit only when one returned chunk contains the checked gold quote."""
    gold_quote = _normalized(question["evidence_quote"])
    gold_source = question["gold_source_id"]
    gold_page = question["gold_page"]
    traced_hits = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        section_ids = metadata.get("section_ids", [])
        if isinstance(section_ids, str):
            section_ids = [section_ids]
        source_match = hit.get("document_id") == gold_source
        page_match = gold_page in section_ids
        evidence_match = gold_quote in _normalized(hit.get("text", ""))
        traced_hits.append(
            {
                **hit,
                "source_match": source_match,
                "page_match": page_match,
                "evidence_match": evidence_match,
                "is_evidence_hit": source_match and page_match and evidence_match,
            }
        )
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "filter": question.get("filter"),
        "gold_source_id": gold_source,
        "gold_page": gold_page,
        "evidence_quote": question["evidence_quote"],
        "hits": traced_hits,
        "hit_at_1": bool(traced_hits and traced_hits[0]["is_evidence_hit"]),
        "evidence_hit_at_3": any(hit["is_evidence_hit"] for hit in traced_hits[:3]),
        "filter_compliant_at_3": (
            all(
                (hit.get("metadata") or {}).get("chapter") == str(question["filter"]["chapter"])
                for hit in traced_hits[:3]
            )
            if question.get("filter") and traced_hits
            else None
        ),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate required counts by strategy and split without inventing labels."""
    summary: dict[str, Any] = {}
    strategies = sorted({trace["strategy"] for trace in traces})
    for strategy in strategies:
        summary[strategy] = {}
        for split in ("dev", "test"):
            subset = [
                trace
                for trace in traces
                if trace["strategy"] == strategy and trace["split"] == split
            ]
            filtered = [trace for trace in subset if trace["filter"]]
            compliant = [
                trace for trace in filtered if trace["filter_compliant_at_3"] is True
            ]
            summary[strategy][split] = {
                "hit_at_1": {
                    "hits": sum(trace["hit_at_1"] for trace in subset),
                    "total": len(subset),
                },
                "evidence_hit_at_3": {
                    "hits": sum(trace["evidence_hit_at_3"] for trace in subset),
                    "total": len(subset),
                },
                "filter_compliance_at_3": (
                    {"hits": len(compliant), "total": len(filtered)}
                    if filtered
                    else None
                ),
            }
    return summary
