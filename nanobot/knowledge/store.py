"""Storage helpers for parsed documents, chunks, and retrieval indexes."""

from __future__ import annotations

from array import array
import json
import math
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from nanobot.knowledge.types import KnowledgeChunk, ParsedDocument


class KnowledgeStore:
    """Persist parsed documents and searchable chunks under the workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        raw_dir: str = "knowledge/raw",
        parsed_dir: str = "knowledge/parsed",
        chunks_dir: str = "knowledge/chunks",
        index_dir: str = "knowledge/index",
        embedding_dim: int = 384,
        vector_index: str = "faiss",
    ):
        self.workspace = workspace
        self.raw_dir = workspace / raw_dir
        self.parsed_dir = workspace / parsed_dir
        self.chunks_dir = workspace / chunks_dir
        self.index_dir = workspace / index_dir
        self.embedding_dim = embedding_dim
        self.vector_index = vector_index
        self.faiss_index_path = self.index_dir / "knowledge.faiss"
        self.faiss_manifest_path = self.index_dir / "knowledge.faiss.json"
        for directory in (self.raw_dir, self.parsed_dir, self.chunks_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.db_path = self.index_dir / "knowledge.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    page INTEGER,
                    heading TEXT,
                    text TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_file)")

    def save_parsed_document(self, parsed: ParsedDocument) -> None:
        target = self.parsed_dir / f"{Path(parsed.source_file).stem}.json"
        payload = {
            "source_file": parsed.source_file,
            "file_type": parsed.file_type,
            "title": parsed.title,
            "parser": parsed.parser,
            "sections": [
                {"text": section.text, "page": section.page, "heading": section.heading}
                for section in parsed.sections
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_raw_file(self, source_path: str | Path) -> Path:
        source = Path(source_path)
        target = self.raw_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def save_chunks(
        self,
        source_file: str,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        target = self.chunks_dir / f"{Path(source_file).stem}.jsonl"
        target.write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks),
            encoding="utf-8",
        )

        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings length must match chunks length")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
            conn.execute("DELETE FROM embeddings WHERE source_file = ?", (source_file,))
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, source_file, file_hash, page, heading, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.source_file,
                        chunk.file_hash,
                        chunk.page,
                        chunk.heading,
                        chunk.text,
                    )
                    for chunk in chunks
                ],
            )
            if embeddings is not None:
                conn.executemany(
                    """
                    INSERT INTO embeddings(chunk_id, source_file, dimension, vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.source_file,
                            len(vector),
                            self._encode_vector(vector),
                        )
                        for chunk, vector in zip(chunks, embeddings)
                    ],
                )
        if embeddings is not None:
            self.rebuild_vector_index()

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
        return self.keyword_search(query=query, top_k=top_k, source_filter=source_filter)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        rows = self._load_chunk_rows(source_filter=source_filter)

        scored: list[tuple[float, dict]] = []
        for row in rows:
            text = str(row["text"]).lower()
            score = sum(text.count(term) for term in terms)
            if score > 0:
                scored.append((float(score), {**row, "keyword_score": float(score)}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []

        if self.vector_index == "faiss" and source_filter is None:
            hits = self._faiss_search(query_embedding=query_embedding, top_k=top_k)
            if hits:
                return hits

        return self._linear_vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter=source_filter,
        )

    def hybrid_search(
        self,
        *,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
        keyword_weight: float = 0.35,
        vector_weight: float = 0.65,
        vector_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        candidate_k = max(top_k, vector_candidates or top_k * 4)
        keyword_hits = self.keyword_search(
            query=query,
            top_k=candidate_k,
            source_filter=source_filter,
        )
        vector_hits = self.vector_search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            source_filter=source_filter,
        )

        keyword_max = max((float(hit.get("keyword_score") or 0.0) for hit in keyword_hits), default=0.0)
        by_id: dict[str, dict[str, Any]] = {}
        for hit in keyword_hits:
            chunk_id = str(hit["chunk_id"])
            row = by_id.setdefault(chunk_id, dict(hit))
            row["keyword_score"] = float(hit.get("keyword_score") or 0.0)
        for hit in vector_hits:
            chunk_id = str(hit["chunk_id"])
            row = by_id.setdefault(chunk_id, dict(hit))
            row["vector_score"] = float(hit.get("vector_score") or 0.0)

        scored: list[dict[str, Any]] = []
        for row in by_id.values():
            keyword_score = float(row.get("keyword_score") or 0.0)
            vector_score = float(row.get("vector_score") or 0.0)
            normalized_keyword = keyword_score / keyword_max if keyword_max > 0 else 0.0
            normalized_vector = max(0.0, vector_score)
            row["score"] = keyword_weight * normalized_keyword + vector_weight * normalized_vector
            if row["score"] > 0:
                scored.append(row)

        scored.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return scored[:top_k]

    def rebuild_vector_index(self) -> None:
        if self.vector_index != "faiss":
            return
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except ImportError:
            return

        rows = self._load_embedding_rows(
            source_filter=None,
            expected_dimension=self.embedding_dim,
        )
        if not rows:
            return

        vectors = [self._decode_vector(row["vector"]) for row in rows]
        if not vectors:
            return

        matrix = np.array(vectors, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(self.faiss_index_path))
        manifest = {
            "dimension": int(matrix.shape[1]),
            "chunk_ids": [row["chunk_id"] for row in rows],
        }
        self.faiss_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _faiss_search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self.faiss_index_path.exists() or not self.faiss_manifest_path.exists():
            return []
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except ImportError:
            return []

        manifest = json.loads(self.faiss_manifest_path.read_text(encoding="utf-8"))
        chunk_ids = list(manifest.get("chunk_ids") or [])
        if not chunk_ids or int(manifest.get("dimension") or 0) != len(query_embedding):
            return []

        index = faiss.read_index(str(self.faiss_index_path))
        query = np.array([query_embedding], dtype="float32")
        scores, positions = index.search(query, min(top_k, len(chunk_ids)))

        hits: list[dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(chunk_ids):
                continue
            row = self._load_chunk_row(str(chunk_ids[position]))
            if row:
                hits.append({**row, "vector_score": float(score)})
        return hits

    def _linear_vector_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        source_filter: str | None,
    ) -> list[dict[str, Any]]:
        rows = self._load_embedding_rows(
            source_filter=source_filter,
            expected_dimension=len(query_embedding),
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        query = self._normalize_vector(query_embedding)
        for row in rows:
            vector = self._decode_vector(row["vector"])
            score = self._dot(query, vector)
            chunk = self._load_chunk_row(str(row["chunk_id"]))
            if chunk:
                scored.append((score, {**chunk, "vector_score": score}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def _load_chunk_rows(self, source_filter: str | None) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT chunk_id, source_file, file_hash, page, heading, text FROM chunks"
            params: list[object] = []
            if source_filter:
                sql += " WHERE source_file = ?"
                params.append(source_filter)
            return [dict(row) for row in conn.execute(sql, params)]

    def _load_chunk_row(self, chunk_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT chunk_id, source_file, file_hash, page, heading, text
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
            return dict(row) if row else None

    def _load_embedding_rows(
        self,
        source_filter: str | None,
        expected_dimension: int | None = None,
    ) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT chunk_id, source_file, dimension, vector FROM embeddings"
            params: list[object] = []
            filters: list[str] = []
            if source_filter:
                filters.append("source_file = ?")
                params.append(source_filter)
            if expected_dimension is not None:
                filters.append("dimension = ?")
                params.append(expected_dimension)
            if filters:
                sql += " WHERE " + " AND ".join(filters)
            return [dict(row) for row in conn.execute(sql, params)]

    @staticmethod
    def _encode_vector(vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def _decode_vector(payload: bytes) -> list[float]:
        values = array("f")
        values.frombytes(payload)
        return list(values)

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))
