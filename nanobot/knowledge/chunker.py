"""Chunking utilities for parsed documents."""

from __future__ import annotations

from hashlib import sha1

from nanobot.knowledge.types import KnowledgeChunk, ParsedDocument


class KnowledgeChunker:
    """Split parsed documents into retrieval-friendly chunks."""

    def __init__(self, max_chunk_chars: int = 1200, chunk_overlap: int = 150):
        self.max_chunk_chars = max(200, max_chunk_chars)
        self.chunk_overlap = max(0, min(chunk_overlap, self.max_chunk_chars // 2))

    def chunk_document(self, doc: ParsedDocument) -> list[KnowledgeChunk]:
        file_hash = sha1(doc.source_file.encode("utf-8")).hexdigest()
        chunks: list[KnowledgeChunk] = []

        for section_index, section in enumerate(doc.sections):
            for part_index, text in enumerate(self._split_text(section.text)):
                chunk_id = f"{file_hash}-{section_index}-{part_index}"
                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_file=doc.source_file,
                    file_hash=file_hash,
                    page=section.page,
                    heading=section.heading,
                    text=text,
                ))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        clean = " ".join(text.split())
        if len(clean) <= self.max_chunk_chars:
            return [clean]

        parts: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + self.max_chunk_chars)
            if end < len(clean):
                split_at = clean.rfind(" ", start, end)
                if split_at > start + 100:
                    end = split_at
            part = clean[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(clean):
                break
            start = end - self.chunk_overlap if self.chunk_overlap else end
        return parts
