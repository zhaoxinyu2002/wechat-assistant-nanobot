import asyncio

from nanobot.agent.tools.knowledge_search import KnowledgeSearchTool
from nanobot.knowledge.service import KnowledgeService


def test_knowledge_search_tool_returns_source_aware_results(tmp_path) -> None:
    source = tmp_path / "guide.txt"
    source.write_text("nanobot can ingest files and build a knowledge base", encoding="utf-8")

    service = KnowledgeService(tmp_path)
    service.ingest_files([str(source)])
    tool = KnowledgeSearchTool(service)

    result = asyncio.run(tool.execute(query="knowledge base", top_k=2))

    assert "source=guide.txt" in result
    assert "score=" in result
    assert "knowledge base" in result
