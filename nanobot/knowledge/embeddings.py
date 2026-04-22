"""Embedding helpers for local knowledge retrieval."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class Embedder(Protocol):
    """Common interface for knowledge-base embedding backends."""

    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_text(self, text: str) -> list[float]:
        """Embed one text."""


@dataclass(slots=True)
class HashingEmbedder:
    """Dependency-free embedding model based on stable feature hashing.

    This is intentionally simple: it gives the knowledge backend a local vector
    path without requiring network calls or heavyweight model downloads. The
    interface leaves room for swapping in sentence-transformers/OpenAI later.
    """

    dimension: int = 384

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        return _l2_normalize(vector)

    def _features(self, text: str) -> list[tuple[str, float]]:
        normalized = text.lower()
        tokens = _TOKEN_RE.findall(normalized)
        features: list[tuple[str, float]] = [(f"tok:{token}", 1.0) for token in tokens]

        for left, right in zip(tokens, tokens[1:]):
            features.append((f"bi:{left} {right}", 1.5))

        # Character n-grams make the fallback less brittle for Chinese text and
        # inflected English without pulling in a tokenizer dependency.
        compact = "".join(tokens)
        for size, weight in ((2, 0.35), (3, 0.25)):
            if len(compact) < size:
                continue
            for index in range(len(compact) - size + 1):
                features.append((f"char{size}:{compact[index:index + size]}", weight))

        return features


@dataclass(slots=True)
class OpenAIEmbedder:
    """OpenAI-compatible embedding backend.

    Works with OpenAI's hosted embeddings and OpenAI-compatible gateways. For
    example, `model="text-embedding-3-small"` with the OpenAI API, or a local
    compatible service that exposes `/v1/embeddings`.
    """

    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str | None = None
    dimension: int = 1536
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI embedding provider requires an API key")
        self.batch_size = max(1, self.batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            kwargs: dict[str, object] = {"model": self.model, "input": batch}
            if self.dimension > 0 and self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimension
            response = client.embeddings.create(**kwargs)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([_l2_normalize(list(item.embedding)) for item in ordered])
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Local sentence-transformers embedding backend.

    Use `model="BAAI/bge-m3"` for a strong multilingual default. The dependency
    is optional so lightweight installs can keep using `HashingEmbedder`.
    """

    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 32
    normalize: bool = True
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embedding models. "
                "Install with `pip install -e .[rag-ml]`."
            ) from exc

        self._model = SentenceTransformer(self.model)
        if hasattr(self._model, "get_embedding_dimension"):
            detected = self._model.get_embedding_dimension()
        else:
            detected = self._model.get_sentence_embedding_dimension()
        if detected:
            self.dimension = int(detected)
        self.batch_size = max(1, self.batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


def build_embedder(
    provider: str,
    *,
    dimension: int,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    batch_size: int = 64,
) -> Embedder:
    """Build an embedding backend from config."""

    key = provider.lower().replace("_", "-")
    if key in {"hashing", "local-hashing"}:
        return HashingEmbedder(dimension=dimension)
    if key in {"openai", "openai-compatible", "text-embedding-3-small"}:
        return OpenAIEmbedder(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            base_url=base_url or None,
            dimension=dimension,
            batch_size=batch_size,
        )
    if key in {"sentence-transformers", "sentence-transformer", "bge-m3", "local"}:
        return SentenceTransformerEmbedder(
            model=model or "BAAI/bge-m3",
            dimension=dimension,
            batch_size=batch_size,
        )
    raise ValueError(f"unsupported embedding provider: {provider}")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
