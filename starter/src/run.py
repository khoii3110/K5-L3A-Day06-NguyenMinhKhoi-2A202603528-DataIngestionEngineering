"""Run the PostgreSQL 17 corpus through the fixed, paragraph, and filtered setups."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenAIEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_jsonl
from .models import Chunk
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
CORPUS_PATH = ROOT / "data" / "postgresql17_units.jsonl"
BENCHMARK_PATH = ROOT / "benchmark" / "postgresql17_questions.jsonl"
RESULTS = STARTER / "results"
STRATEGIES = ("A_fixed", "B_paragraph", "C_paragraph_filter")


def _read_questions(path: Path, pages: list[Any]) -> list[dict[str, Any]]:
    questions = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw in enumerate(file, start=1):
            if not raw.strip():
                continue
            try:
                question = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            required = {
                "id",
                "split",
                "question",
                "expected_answer",
                "gold_source_id",
                "gold_page",
                "evidence_quote",
                "filter",
            }
            missing = required - question.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {', '.join(sorted(missing))}"
                )
            matching_sources = [
                page
                for page in pages
                if page.document_id == question["gold_source_id"]
                and question["gold_page"] in page.metadata["section_ids"]
            ]
            if not any(
                question["evidence_quote"] in page.text for page in matching_sources
            ):
                raise ValueError(
                    f"{question['id']}: gold evidence is not present in the cited source section."
                )
            questions.append(question)
    if len(questions) < 20:
        raise ValueError(f"Expected at least 20 benchmark questions; found {len(questions)}.")
    if sum(question["split"] == "dev" for question in questions) < 15:
        raise ValueError("Benchmark needs at least 15 dev questions.")
    if sum(question["split"] == "test" for question in questions) < 5:
        raise ValueError("Benchmark needs at least 5 test questions.")
    return questions


def _tag_strategy(chunks: list[Chunk], strategy: str) -> list[Chunk]:
    import hashlib

    return [
        Chunk(
            chunk_id=hashlib.sha256(f"{strategy}:{chunk.chunk_id}".encode()).hexdigest(),
            document_id=chunk.document_id,
            source=chunk.source,
            page=chunk.page,
            text=chunk.text,
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    ]


def _chunk_stats(chunks: list[Chunk]) -> dict[str, int | float]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_characters": sum(lengths),
        "min_chunk_characters": min(lengths, default=0),
        "max_chunk_characters": max(lengths, default=0),
        "mean_chunk_characters": round(sum(lengths) / len(lengths), 2) if lengths else 0,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_audit_templates(
    traces: list[dict[str, Any]],
    pages: list[Any],
    chunks_by_strategy: dict[str, list[Chunk]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "grounding_audit.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["strategy", "question_id", "top3_chunk_ids", "grounding_score_0_1_2", "reason"]
        )
        for trace in traces:
            if trace["split"] == "test":
                writer.writerow(
                    [
                        trace["strategy"],
                        trace["question_id"],
                        ";".join(hit["chunk_id"] for hit in trace["hits"][:3]),
                        "",
                        "",
                    ]
                )

    with (RESULTS / "metadata_audit.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "strategy",
                "question_id",
                "rank",
                "chunk_id",
                "source",
                "page_unit_id",
                "section_ids",
                "chapter",
                "expected_filter",
                "metadata_correct",
                "error_note",
            ]
        )
        for trace in traces:
            if trace["split"] != "test":
                continue
            for rank, hit in enumerate(trace["hits"][:3], start=1):
                metadata = hit.get("metadata") or {}
                writer.writerow(
                    [
                        trace["strategy"],
                        trace["question_id"],
                        rank,
                        hit["chunk_id"],
                        hit["source"],
                        hit["page"],
                        ";".join(metadata.get("section_ids", [])),
                        metadata.get("chapter", ""),
                        json.dumps(trace["filter"], ensure_ascii=False) if trace["filter"] else "",
                        "",
                        "",
                    ]
                )

    ordered_pages = pages
    sample_indexes = sorted(
        {round(index * (len(ordered_pages) - 1) / 9) for index in range(10)}
    )
    sampled_pages = [ordered_pages[index] for index in sample_indexes]
    with (RESULTS / "coherence_audit.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source_unit_id",
                "section_ids",
                "strategy",
                "chunk_id",
                "chunk_text",
                "coherence_score_0_1_2",
                "reason",
            ]
        )
        for page in sampled_pages:
            midpoint = len(page.text) // 2
            for strategy, chunks in chunks_by_strategy.items():
                candidates = [
                    chunk
                    for chunk in chunks
                    if chunk.page == page.page
                    and chunk.metadata["char_start"] <= midpoint < chunk.metadata["char_end"]
                ]
                chunk = candidates[0] if candidates else None
                writer.writerow(
                    [
                        page.page,
                        ";".join(page.metadata["section_ids"]),
                        strategy,
                        chunk.chunk_id if chunk else "",
                        chunk.text if chunk else "",
                        "",
                        "",
                    ]
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and chunk counts without connecting to PostgreSQL or an embedding API",
    )
    arguments = parser.parse_args(argv)
    load_dotenv(STARTER / ".env")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pages = load_jsonl(CORPUS_PATH)
    if len(pages) < 30:
        raise ValueError(f"Expected at least 30 source units; found {len(pages)}.")
    questions = _read_questions(BENCHMARK_PATH, pages)

    chunking_times: dict[str, float] = {}
    chunking_started = time.perf_counter()
    chunks_by_strategy: dict[str, list[Chunk]] = {
        "A_fixed": [
            chunk
            for page in pages
            for chunk in fixed_size(page, size=1_000, overlap=100)
        ],
    }
    chunking_times["A_fixed"] = time.perf_counter() - chunking_started
    chunking_started = time.perf_counter()
    chunks_by_strategy["B_paragraph"] = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=1_000)
    ]
    chunking_times["B_paragraph"] = time.perf_counter() - chunking_started
    chunking_started = time.perf_counter()
    chunks_by_strategy["C_paragraph_filter"] = _tag_strategy(
        chunks_by_strategy["B_paragraph"], "C_paragraph_filter"
    )
    chunking_times["C_paragraph_filter"] = time.perf_counter() - chunking_started
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "source_units": len(pages),
                    "questions": len(questions),
                    "strategies": {
                        strategy: _chunk_stats(chunks)
                        for strategy, chunks in chunks_by_strategy.items()
                    },
                    "database_connected": False,
                    "embedding_api_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required; configure starter/.env.")
    embedder = OpenAIEmbedder()
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    database_versions = store.database_versions()

    vectors_by_strategy: dict[str, list[list[float]]] = {}
    strategy_metrics: dict[str, Any] = {}
    for strategy in STRATEGIES:
        started = time.perf_counter()
        chunks = chunks_by_strategy[strategy]
        token_start = embedder.total_tokens
        if strategy == "C_paragraph_filter":
            vectors = vectors_by_strategy["B_paragraph"]
        else:
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            vectors_by_strategy[strategy] = vectors
        document_tokens = embedder.total_tokens - token_start
        store.upsert(strategy, chunks, vectors)
        strategy_metrics[strategy] = {
            **_chunk_stats(chunks),
            "chunking_seconds": round(chunking_times[strategy], 3),
            "document_embedding_tokens": document_tokens,
            "embedding_reused_from": "B_paragraph" if strategy == "C_paragraph_filter" else None,
            "ingest_seconds": round(time.perf_counter() - started, 3),
        }

    query_token_start = embedder.total_tokens
    query_vectors = embedder.embed_documents([question["question"] for question in questions])
    query_tokens = embedder.total_tokens - query_token_start
    traces = []
    search_metrics: dict[str, dict[str, float | int]] = {}
    for strategy in STRATEGIES:
        search_started = time.perf_counter()
        for question, query_vector in zip(questions, query_vectors):
            filters = (
                question.get("filter")
                if strategy == "C_paragraph_filter" and question.get("filter")
                else None
            )
            hits = store.search(strategy, query_vector, top_k=3, filters=filters)
            traces.append(
                {
                    "strategy": strategy,
                    **evaluate_query(question, hits),
                }
            )
        elapsed = time.perf_counter() - search_started
        search_metrics[strategy] = {
            "total_seconds": round(elapsed, 3),
            "mean_seconds_per_query": round(elapsed / len(questions), 6),
        }

    _write_jsonl(RESULTS / "top3_trace.jsonl", traces)
    _write_json(RESULTS / "retrieval_summary.json", summarize(traces))
    _write_audit_templates(traces, pages, chunks_by_strategy)

    cost_rate = os.getenv("EMBEDDING_USD_PER_1M_TOKENS")
    total_tokens = embedder.total_tokens
    run_metadata = {
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "benchmark": str(BENCHMARK_PATH.relative_to(ROOT)),
        "source_units": len(pages),
        "questions": len(questions),
        "strategies": strategy_metrics,
        "embedding_model": embedder.model_name,
        "embedding_provider": embedder.provider,
        "embedding_dimensions": embedder.dimensions,
        "embedding_api_requests": embedder.requests,
        "document_embedding_tokens": sum(
            value["document_embedding_tokens"] for value in strategy_metrics.values()
        ),
        "query_embedding_tokens": query_tokens,
        "total_embedding_tokens_including_queries": total_tokens,
        "embedding_cost_usd": (
            round(total_tokens * float(cost_rate) / 1_000_000, 6)
            if cost_rate is not None
            else None
        ),
        "embedding_cost_note": (
            "Estimated using EMBEDDING_USD_PER_1M_TOKENS."
            if cost_rate is not None
            else "Not estimated; set EMBEDDING_USD_PER_1M_TOKENS from the provider's current pricing."
        ),
        "vector_store": "PostgreSQL + pgvector",
        "database_versions": database_versions,
        "python_version": platform.python_version(),
        "library_versions": {
            "psycopg": importlib.metadata.version("psycopg"),
            "pypdf": importlib.metadata.version("pypdf"),
            "python-dotenv": importlib.metadata.version("python-dotenv"),
        },
        "search_metric": "cosine distance",
        "search_type": "exact (no approximate vector index)",
        "search_metrics": search_metrics,
    }
    _write_json(RESULTS / "run_metadata.json", run_metadata)
    print(f"Run complete: {len(traces)} traces saved under {RESULTS}.")
    if cost_rate is None:
        print(run_metadata["embedding_cost_note"])


if __name__ == "__main__":
    main()
