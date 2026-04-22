"""Optional reranking backends for knowledge retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nanobot.knowledge.types import SearchHit


class Reranker(Protocol):
    """Common interface for retrieval rerankers."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return hits reordered by relevance."""


@dataclass(slots=True)
class NoopReranker:
    """Keep retrieval order unchanged."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]


@dataclass(slots=True)
class SentenceTransformerReranker:
    """Cross-encoder reranker, suitable for `BAAI/bge-reranker-*` models."""

    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 16
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with `pip install -e .[rag-ml]`."
            ) from exc
        self._model = CrossEncoder(self.model)
        self.batch_size = max(1, self.batch_size)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored = [
            SearchHit(chunk=hit.chunk, score=float(score))
            for hit, score in zip(hits, scores)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]


def build_reranker(provider: str, *, model: str = "", batch_size: int = 16) -> Reranker:
    """Build a reranker backend from config."""

    key = provider.lower().replace("_", "-")
    if key in {"", "none", "noop", "disabled"}:
        return NoopReranker()
    if key in {"sentence-transformers", "sentence-transformer", "bge-reranker", "bge"}:
        return SentenceTransformerReranker(
            model=model or "BAAI/bge-reranker-base",
            batch_size=batch_size,
        )
    raise ValueError(f"unsupported reranker provider: {provider}")
