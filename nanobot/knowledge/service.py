"""High-level ingestion and retrieval service."""

from __future__ import annotations

from pathlib import Path

from nanobot.knowledge.chunker import KnowledgeChunker
from nanobot.knowledge.embeddings import Embedder, build_embedder
from nanobot.knowledge.parser import DocumentParser
from nanobot.knowledge.reranker import Reranker, build_reranker
from nanobot.knowledge.store import KnowledgeStore
from nanobot.knowledge.types import IngestResult, KnowledgeChunk, SearchHit


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
        max_file_bytes: int = 30 * 1024 * 1024,
        max_chunks_per_file: int = 1000,
        max_chunk_chars: int = 1200,
        chunk_overlap: int = 150,
        chunk_strategy: str = "recursive",
        chunk_include_metadata: bool = True,
        embedding_provider: str = "hashing",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        embedding_dim: int = 384,
        embedding_batch_size: int = 64,
        vector_index: str = "faiss",
        retrieval_mode: str = "hybrid",
        keyword_weight: float = 0.35,
        vector_weight: float = 0.65,
        reranker_provider: str = "none",
        reranker_model: str = "",
        reranker_top_k: int = 20,
        reranker_batch_size: int = 16,
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
        self.max_file_bytes = max(1, max_file_bytes)
        self.max_chunks_per_file = max(1, max_chunks_per_file)
        self.store = KnowledgeStore(
            workspace,
            raw_dir=raw_dir,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            embedding_dim=embedding_dim,
            vector_index=vector_index,
        )
        self.embedding_provider = embedding_provider
        self.retrieval_mode = retrieval_mode
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.reranker_top_k = max(1, reranker_top_k)
        self.embedder: Embedder = build_embedder(
            provider=embedding_provider,
            dimension=embedding_dim,
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            batch_size=embedding_batch_size,
        )
        if self.store.embedding_dim != self.embedder.dimension:
            self.store.embedding_dim = self.embedder.dimension
        self.reranker: Reranker = build_reranker(
            reranker_provider,
            model=reranker_model,
            batch_size=reranker_batch_size,
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
            strategy=chunk_strategy,
            include_metadata=chunk_include_metadata,
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
            size = path.stat().st_size
            if size > self.max_file_bytes:
                raise ValueError(
                    f"file too large for automatic ingestion: {size} bytes "
                    f"(limit {self.max_file_bytes} bytes)"
                )
            self.store.save_raw_file(path)
            parsed = self.parser.parse_file(path)
            chunks = self.chunker.chunk_document(parsed)
            if not chunks:
                raise ValueError("no chunks generated from parsed document")
            if len(chunks) > self.max_chunks_per_file:
                raise ValueError(
                    f"too many chunks generated: {len(chunks)} "
                    f"(limit {self.max_chunks_per_file}); increase maxChunksPerFile "
                    "or split the document"
                )
            embeddings = self.embed_chunks(chunks)
            self.store.save_parsed_document(parsed)
            self.store.save_chunks(parsed.source_file, chunks, embeddings=embeddings)
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

    def embed_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if embeddings:
            self.store.embedding_dim = len(embeddings[0])
        return embeddings

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        retrieval_mode: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        if not query.strip():
            return []

        mode = (retrieval_mode or self.retrieval_mode).lower()
        candidate_top_k = max(top_k, self.reranker_top_k)
        if mode == "keyword":
            rows = self.store.keyword_search(query=query, top_k=candidate_top_k, source_filter=source_filter)
        elif mode == "vector":
            query_embedding = self.embedder.embed_text(query)
            if query_embedding:
                self.store.embedding_dim = len(query_embedding)
            rows = self.store.vector_search(
                query_embedding=query_embedding,
                top_k=candidate_top_k,
                source_filter=source_filter,
            )
        else:
            query_embedding = self.embedder.embed_text(query)
            if query_embedding:
                self.store.embedding_dim = len(query_embedding)
            rows = self.store.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=candidate_top_k,
                source_filter=source_filter,
                keyword_weight=self.keyword_weight,
                vector_weight=self.vector_weight,
            )

        hits: list[SearchHit] = []
        for row in rows:
            score = float(row.get("score") or row.get("vector_score") or row.get("keyword_score") or 0.0)
            if score < min_score:
                continue
            hits.append(SearchHit(
                chunk=KnowledgeChunk(
                    chunk_id=row["chunk_id"],
                    source_file=row["source_file"],
                    file_hash=row["file_hash"],
                    page=row["page"],
                    heading=row["heading"],
                    text=row["text"],
                ),
                score=score,
            ))
        return self.reranker.rerank(query, hits, top_k)
