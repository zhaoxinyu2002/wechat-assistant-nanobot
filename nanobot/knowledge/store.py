"""Storage helpers for parsed documents and chunks."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

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
    ):
        self.workspace = workspace
        self.raw_dir = workspace / raw_dir
        self.parsed_dir = workspace / parsed_dir
        self.chunks_dir = workspace / chunks_dir
        self.index_dir = workspace / index_dir
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

    def save_chunks(self, source_file: str, chunks: list[KnowledgeChunk]) -> None:
        target = self.chunks_dir / f"{Path(source_file).stem}.jsonl"
        target.write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks),
            encoding="utf-8",
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
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

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT chunk_id, source_file, file_hash, page, heading, text FROM chunks"
            params: list[object] = []
            if source_filter:
                sql += " WHERE source_file = ?"
                params.append(source_filter)
            rows = [dict(row) for row in conn.execute(sql, params)]

        scored: list[tuple[float, dict]] = []
        for row in rows:
            text = str(row["text"]).lower()
            score = sum(text.count(term) for term in terms)
            if score > 0:
                scored.append((float(score), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]
