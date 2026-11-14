"""Persistent pgvector store with exact cosine search and parameterized filters."""
from typing import Any
import math

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required.")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table_name = f"day6_chunks_{dimensions}"
        self.chapter_index_name = f"day6_chunks_{dimensions}_strategy_chapter_idx"

    def create_schema(self) -> None:
        """Create a cosine-searchable table; no approximate index means exact search."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        chunk_id TEXT PRIMARY KEY,
                        strategy TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding VECTOR({self.dimensions}) NOT NULL
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.chapter_index_name}
                    ON {self.table_name} (strategy, (metadata ->> 'chapter'))
                    """
                )

    def database_versions(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_setting('server_version'),
                           (SELECT extversion FROM pg_extension WHERE extname = 'vector')
                    """
                )
                row = cursor.fetchone()
        if row is None or row[1] is None:
            raise RuntimeError("PostgreSQL is connected but pgvector is not installed.")
        return {"postgresql": row[0], "pgvector": row[1]}

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy atomically, removing stale chunks and duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")
        rows = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"{chunk.chunk_id} has {len(vector)} dimensions; expected {self.dimensions}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"{chunk.chunk_id} has a non-finite embedding value.")
            rows.append(
                (
                    chunk.chunk_id,
                    strategy,
                    chunk.document_id,
                    chunk.source,
                    str(chunk.page),
                    chunk.text,
                    Jsonb(chunk.metadata),
                    "[" + ",".join(repr(float(value)) for value in vector) + "]",
                )
            )
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE strategy = %s", (strategy,)
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table_name}
                        (chunk_id, strategy, document_id, source, page, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    rows,
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run exact cosine search and apply the optional chapter filter in SQL."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query has {len(query_vector)} dimensions; expected {self.dimensions}."
            )
        if not all(math.isfinite(value) for value in query_vector):
            raise ValueError("Query vector contains a non-finite value.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        filters = filters or {}
        unsupported = set(filters) - {"chapter"}
        if unsupported:
            raise ValueError(f"Unsupported metadata filter(s): {', '.join(sorted(unsupported))}")
        where = "strategy = %s"
        parameters: list[Any] = [strategy]
        if "chapter" in filters:
            where += " AND metadata ->> 'chapter' = %s"
            parameters.append(str(filters["chapter"]))
        vector_literal = "[" + ",".join(repr(float(value)) for value in query_vector) + "]"
        parameters = [vector_literal, *parameters, vector_literal, top_k]
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source, page, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self.table_name}
                    WHERE {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    parameters,
                )
                rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
