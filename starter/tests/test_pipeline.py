import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.chunking import by_paragraph, fixed_size
from src.embeddings import OpenAIEmbedder
from src.evaluation import evaluate_query, summarize
from src.ingestion import load_jsonl, normalize_text
from src.models import DocumentPage
from src.run import _read_questions


ROOT = Path(__file__).resolve().parents[2]


class IngestionAndChunkingTests(unittest.TestCase):
    def test_normalize_preserves_paragraph_boundaries(self) -> None:
        self.assertEqual(normalize_text("e\u0301  \r\n\r\nx\t"), "é\n\nx")

    def test_fixed_size_tracks_overlap_and_offsets(self) -> None:
        page = DocumentPage("doc", "https://example.test", "unit-1", "x" * 1_100)
        chunks = fixed_size(page, size=500, overlap=50)
        self.assertEqual([len(chunk.text) for chunk in chunks], [500, 500, 200])
        self.assertEqual(chunks[1].metadata["char_start"], 450)
        self.assertEqual(chunks[0].metadata["char_end"], 500)

    def test_paragraph_chunking_respects_limit(self) -> None:
        page = DocumentPage("doc", "https://example.test", "unit-1", "A" * 600 + "\n\n" + "B" * 600)
        chunks = by_paragraph(page, max_size=800)
        self.assertEqual([chunk.text for chunk in chunks], ["A" * 600, "B" * 600])
        self.assertTrue(all(chunk.metadata["char_start"] < chunk.metadata["char_end"] for chunk in chunks))

    def test_real_model_requires_credentials_without_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIEmbedder()

    def test_openrouter_key_selects_openrouter_model_and_endpoint(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENAI_API_KEY": "",
                "EMBEDDING_MODEL": "",
            },
            clear=True,
        ):
            embedder = OpenAIEmbedder()
        self.assertEqual(embedder.provider, "openrouter")
        self.assertEqual(embedder.model_name, "openai/text-embedding-3-small")
        self.assertEqual(embedder.dimensions, 1536)
        self.assertEqual(embedder.base_url, "https://openrouter.ai/api/v1")


class BenchmarkTests(unittest.TestCase):
    def test_corpus_and_gold_evidence_are_valid(self) -> None:
        pages = load_jsonl(ROOT / "data" / "postgresql17_units.jsonl")
        questions = _read_questions(ROOT / "benchmark" / "postgresql17_questions.jsonl", pages)
        self.assertGreaterEqual(len(pages), 30)
        self.assertEqual(sum(question["split"] == "dev" for question in questions), 15)
        self.assertEqual(sum(question["split"] == "test" for question in questions), 5)

    def test_exact_evidence_hit_is_distinct_from_manual_labels(self) -> None:
        question = {
            "id": "q1",
            "split": "test",
            "question": "Question?",
            "expected_answer": "Answer.",
            "filter": {"chapter": "15"},
            "gold_source_id": "doc",
            "gold_page": "section-1",
            "evidence_quote": "evidence sentence",
        }
        unrelated = {
            "chunk_id": "c1",
            "document_id": "doc",
            "metadata": {"section_ids": ["section-2"], "chapter": "15"},
            "text": "evidence sentence",
        }
        matching = {
            "chunk_id": "c2",
            "document_id": "doc",
            "metadata": {"section_ids": ["section-1"], "chapter": "15"},
            "text": "Some evidence sentence follows.",
        }
        trace = evaluate_query(question, [unrelated, matching])
        self.assertFalse(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])
        self.assertTrue(trace["filter_compliant_at_3"])
        summary = summarize([{"strategy": "C", **trace}])
        self.assertEqual(summary["C"]["test"]["evidence_hit_at_3"], {"hits": 1, "total": 1})


if __name__ == "__main__":
    unittest.main()
