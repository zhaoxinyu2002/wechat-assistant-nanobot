"""High-level ingestion and retrieval service."""

from __future__ import annotations

from pathlib import Path

from nanobot.knowledge.chunker import KnowledgeChunker
from nanobot.knowledge.parser import DocumentParser
from nanobot.knowledge.store import KnowledgeStore
from nanobot.knowledge.types import IngestResult, SearchHit, KnowledgeChunk


class KnowledgeService:
    """Coordinate parsing, chunking, persistence, and retrieval."""

    SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}

    def __init__(
        self,
        workspace: Path,
        *,
        enabled: bool = True,
        auto_ingest_from_media: bool = True,
        raw_dir: str = "knowledge/raw",
        parsed_dir: str = "knowledge/parsed",
        chunks_dir: str = "knowledge/chunks",
        index_dir: str = "knowledge/index",
        max_chunk_chars: int = 1200,
        chunk_overlap: int = 150,
        parser_pdf: str = "mineru",
        mineru_command: str = "",
        mineru_mode: str = "agent",
        mineru_base_url: str = "https://mineru.net",
        mineru_api_token: str = "",
        mineru_model_version: str = "vlm",
        mineru_language: str = "ch",
        mineru_enable_table: bool = True,
        mineru_enable_formula: bool = True,
        mineru_is_ocr: bool = True,
        mineru_page_range: str | None = None,
        mineru_timeout_s: int = 300,
        mineru_poll_interval_s: int = 3,
    ):
        self.workspace = workspace
        self.enabled = enabled
        self.auto_ingest_from_media = auto_ingest_from_media
        self.store = KnowledgeStore(
            workspace,
            raw_dir=raw_dir,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
        )
        self.parser = DocumentParser(
            pdf_parser=parser_pdf,
            command=mineru_command,
            mode=mineru_mode,
            base_url=mineru_base_url,
            api_token=mineru_api_token,
            model_version=mineru_model_version,
            language=mineru_language,
            enable_table=mineru_enable_table,
            enable_formula=mineru_enable_formula,
            is_ocr=mineru_is_ocr,
            page_range=mineru_page_range,
            timeout_s=mineru_timeout_s,
            poll_interval_s=mineru_poll_interval_s,
        )
        self.chunker = KnowledgeChunker(
            max_chunk_chars=max_chunk_chars,
            chunk_overlap=chunk_overlap,
        )

    def should_ingest(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.SUPPORTED_SUFFIXES

    def ingest_files(self, paths: list[str]) -> list[IngestResult]:
        if not self.enabled:
            return []

        results: list[IngestResult] = []
        seen: set[str] = set()
        for raw_path in paths:
            path = str(Path(raw_path))
            if path in seen:
                continue
            seen.add(path)
            if not self.should_ingest(path):
                continue
            results.append(self._ingest_file(Path(path)))
        return results

    def _ingest_file(self, path: Path) -> IngestResult:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            self.store.save_raw_file(path)
            parsed = self.parser.parse_file(path)
            chunks = self.chunker.chunk_document(parsed)
            if not chunks:
                raise ValueError("no chunks generated from parsed document")
            self.store.save_parsed_document(parsed)
            self.store.save_chunks(parsed.source_file, chunks)
            return IngestResult(
                path=str(path),
                status="ok",
                chunks_created=len(chunks),
                parser=parsed.parser,
            )
        except Exception as exc:
            return IngestResult(
                path=str(path),
                status="error",
                error=str(exc),
            )

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[SearchHit]:
        rows = self.store.search(query=query, top_k=top_k, source_filter=source_filter)
        hits: list[SearchHit] = []
        for row in rows:
            hits.append(SearchHit(
                chunk=KnowledgeChunk(
                    chunk_id=row["chunk_id"],
                    source_file=row["source_file"],
                    file_hash=row["file_hash"],
                    page=row["page"],
                    heading=row["heading"],
                    text=row["text"],
                ),
                score=1.0,
            ))
        return hits
