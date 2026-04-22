"""Chunking utilities for parsed documents."""

from __future__ import annotations

from hashlib import sha1

from nanobot.knowledge.types import KnowledgeChunk, ParsedDocument


class KnowledgeChunker:
    """Split parsed documents into retrieval-friendly chunks."""

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        chunk_overlap: int = 150,
        strategy: str = "recursive",
        include_metadata: bool = True,
    ):
        self.max_chunk_chars = max(200, max_chunk_chars)
        self.chunk_overlap = max(0, min(chunk_overlap, self.max_chunk_chars // 2))
        self.strategy = strategy.lower().replace("_", "-")
        self.include_metadata = include_metadata

    def chunk_document(self, doc: ParsedDocument) -> list[KnowledgeChunk]:
        file_hash = sha1(doc.source_file.encode("utf-8")).hexdigest()
        chunks: list[KnowledgeChunk] = []
        sections = self._sections_for_strategy(doc)

        for section_index, section in enumerate(sections):
            text = self._format_section_text(section.heading, section.text)
            parts = [text] if self.strategy == "section" else self._split_text(text)
            for part_index, part in enumerate(parts):
                chunk_id = f"{file_hash}-{section_index}-{part_index}"
                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    source_file=doc.source_file,
                    file_hash=file_hash,
                    page=section.page,
                    heading=section.heading,
                    text=part,
                ))
        return chunks

    def _sections_for_strategy(self, doc: ParsedDocument):
        if self.strategy != "page":
            return doc.sections

        from nanobot.knowledge.types import ParsedSection

        grouped: dict[int | None, list[str]] = {}
        headings: dict[int | None, str | None] = {}
        for section in doc.sections:
            grouped.setdefault(section.page, []).append(section.text)
            if section.heading and section.page not in headings:
                headings[section.page] = section.heading
        return [
            ParsedSection(text="\n\n".join(texts), page=page, heading=headings.get(page))
            for page, texts in grouped.items()
        ]

    def _format_section_text(self, heading: str | None, text: str) -> str:
        if not self.include_metadata or not heading:
            return text
        if text.strip().startswith(heading.strip()):
            return text
        return f"{heading}\n\n{text}"

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
