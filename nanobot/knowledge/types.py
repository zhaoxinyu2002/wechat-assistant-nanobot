"""Datatypes for the knowledge-base pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedSection:
    """One normalized text section extracted from a document."""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    """A normalized parsed document."""

    source_file: str
    file_type: str
    title: str | None = None
    sections: list[ParsedSection] = field(default_factory=list)
    parser: str = "basic"


@dataclass(slots=True)
class KnowledgeChunk:
    """A searchable text chunk with source metadata."""

    chunk_id: str
    source_file: str
    file_hash: str
    text: str
    page: int | None = None
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    """One retrieval result."""

    chunk: KnowledgeChunk
    score: float


@dataclass(slots=True)
class IngestResult:
    """Outcome of ingesting one file into the knowledge base."""

    path: str
    status: str
    chunks_created: int = 0
    parser: str | None = None
    error: str | None = None
