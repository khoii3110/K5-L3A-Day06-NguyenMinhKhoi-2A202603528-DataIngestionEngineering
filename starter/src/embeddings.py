"""OpenAI-compatible embedding API adapter; never falls back to mock vectors."""
import json
import math
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenAIEmbedder:
    """Use one real OpenAI or OpenRouter embedding model for documents and queries."""

    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str | None = None) -> None:
        self.provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not self.provider:
            if openrouter_key and not openai_key:
                self.provider = "openrouter"
            elif openai_key and not openrouter_key:
                self.provider = "openai"
            elif openrouter_key and openai_key:
                raise RuntimeError(
                    "Both API keys are configured; set EMBEDDING_PROVIDER to openrouter or openai."
                )
            else:
                raise RuntimeError(
                    "Set OPENROUTER_API_KEY or OPENAI_API_KEY to use a real embedding model."
                )
        if self.provider not in {"openrouter", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be openrouter or openai.")
        self.api_key = openrouter_key if self.provider == "openrouter" else openai_key
        if not self.api_key:
            raise RuntimeError(f"The API key for {self.provider} is not configured.")
        default_model = (
            "openai/text-embedding-3-small"
            if self.provider == "openrouter"
            else "text-embedding-3-small"
        )
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or default_model
        if self.provider == "openrouter" and "/" not in self.model_name:
            raise ValueError(
                "OpenRouter model names must include the provider prefix, "
                "for example openai/text-embedding-3-small."
            )
        if self.provider == "openai" and "/" in self.model_name:
            raise ValueError("Use the plain model name with the direct OpenAI provider.")
        configured_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        dimensions_key = (
            self.model_name.removeprefix("openai/")
            if self.provider == "openrouter"
            else self.model_name
        )
        self.dimensions = int(
            configured_dimensions or self.KNOWN_DIMENSIONS.get(dimensions_key, 0)
        )
        if self.dimensions <= 0:
            raise ValueError(
                "Set EMBEDDING_DIMENSIONS for an embedding model with an unknown dimension."
            )
        self.api_dimensions = (
            self.dimensions
            if configured_dimensions and dimensions_key.startswith("text-embedding-3-")
            else None
        )
        if (
            configured_dimensions
            and dimensions_key in self.KNOWN_DIMENSIONS
            and dimensions_key not in {"text-embedding-3-small", "text-embedding-3-large"}
            and self.dimensions != self.KNOWN_DIMENSIONS[dimensions_key]
        ):
            raise ValueError(
                f"{self.model_name} always returns {self.KNOWN_DIMENSIONS[dimensions_key]} dimensions."
            )
        default_base_url = (
            "https://openrouter.ai/api/v1"
            if self.provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        base_url_name = "OPENROUTER_BASE_URL" if self.provider == "openrouter" else "OPENAI_BASE_URL"
        self.base_url = os.getenv(base_url_name, default_base_url).rstrip("/")
        self.total_tokens = 0
        self.requests = 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body: dict[str, str | list[str] | int] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.api_dimensions is not None:
            body["dimensions"] = self.api_dimensions
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **(
                    {"HTTP-Referer": "https://github.com/khoii3110/K5-L3A-Day06-NguyenMinhKhoi-2A202603528-DataIngestionEngineering"}
                    if self.provider == "openrouter"
                    else {}
                ),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1_000]
            raise RuntimeError(
                f"{self.provider} embedding API returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"{self.provider} embedding API request failed: {error.reason}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{self.provider} returned invalid JSON: {error}") from error

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Embedding API response did not contain one vector per input.")
        data.sort(key=lambda item: item.get("index", -1))
        vectors = [item.get("embedding") for item in data]
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Embedding {index} has dimension {len(vector) if isinstance(vector, list) else 'unknown'}; "
                    f"expected {self.dimensions}."
                )
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
                raise RuntimeError(f"Embedding {index} contains a non-finite value.")
        usage = payload.get("usage", {})
        self.total_tokens += int(usage.get("prompt_tokens", 0))
        self.requests += 1
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for offset in range(0, len(texts), 64):
            vectors.extend(self._embed(texts[offset : offset + 64]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]
