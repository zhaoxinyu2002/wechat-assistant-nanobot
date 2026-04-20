"""Knowledge-base retrieval tool."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.knowledge.service import KnowledgeService


class KnowledgeSearchTool(Tool):
    """Retrieve relevant chunks from the local knowledge base."""

    def __init__(self, service: KnowledgeService):
        self.service = service

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "Search the local knowledge base before answering questions about user-uploaded "
            "documents. Returns source-aware evidence snippets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user question or search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of evidence snippets to return",
                    "minimum": 1,
                    "maximum": 10,
                },
                "source_filter": {
                    "type": ["string", "null"],
                    "description": "Optional file name filter",
                },
                "retrieval_mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "description": "Retrieval backend to use. Defaults to hybrid.",
                },
                "min_score": {
                    "type": "number",
                    "description": "Drop evidence below this retrieval score",
                    "minimum": 0,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str | None = None,
        top_k: int = 5,
        source_filter: str | None = None,
        retrieval_mode: str | None = None,
        min_score: float = 0.0,
        **_: Any,
    ) -> str:
        if not query:
            return "Error: query is required"
        hits = self.service.search(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
            retrieval_mode=retrieval_mode,
            min_score=min_score,
        )
        if not hits:
            return "No relevant knowledge found."

        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            meta: list[str] = [f"source={chunk.source_file}"]
            if chunk.page is not None:
                meta.append(f"page={chunk.page}")
            if chunk.heading:
                meta.append(f"heading={chunk.heading}")
            meta.append(f"score={hit.score:.3f}")
            lines.append(f"[{index}] {' '.join(meta)}")
            lines.append(chunk.text)
            lines.append("")
        return "\n".join(lines).strip()
