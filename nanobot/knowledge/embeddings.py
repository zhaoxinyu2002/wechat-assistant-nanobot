"""Embedding helpers for local knowledge retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


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


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
