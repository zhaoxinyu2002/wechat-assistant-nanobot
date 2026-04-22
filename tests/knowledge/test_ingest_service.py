import sqlite3
from zipfile import ZipFile

from nanobot.knowledge.service import KnowledgeService


def test_ingest_service_saves_chunks_and_supports_search(tmp_path) -> None:
    source = tmp_path / "notes.md"
    source.write_text(
        "# Transformer\n\nAttention is all you need.\n\nTransformers use self-attention.",
        encoding="utf-8",
    )

    service = KnowledgeService(tmp_path)
    results = service.ingest_files([str(source)])

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].chunks_created >= 1

    hits = service.search("self-attention", top_k=3)
    assert hits
    assert hits[0].chunk.source_file == "notes.md"
    assert "self-attention" in hits[0].chunk.text


def test_ingest_service_persists_embeddings_and_supports_vector_search(tmp_path) -> None:
    source = tmp_path / "rag.txt"
    source.write_text(
        "Hybrid retrieval combines keyword matching with vector similarity.",
        encoding="utf-8",
    )

    service = KnowledgeService(tmp_path)
    results = service.ingest_files([str(source)])

    assert results[0].status == "ok"
    with sqlite3.connect(tmp_path / "knowledge" / "index" / "knowledge.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count >= 1

    hits = service.search("vector similarity", top_k=1, retrieval_mode="vector")

    assert hits
    assert hits[0].score >= 0
    assert hits[0].chunk.source_file == "rag.txt"


def test_ingest_service_saves_raw_file_and_parses_docx(tmp_path) -> None:
    source = tmp_path / "guide.docx"
    with ZipFile(source, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>',
        )
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Knowledge systems need citations.</w:t></w:r></w:p></w:body></w:document>',
        )

    service = KnowledgeService(tmp_path)
    results = service.ingest_files([str(source)])

    assert results[0].status == "ok"
    assert (tmp_path / "knowledge" / "raw" / "guide.docx").exists()
    hits = service.search("citations", top_k=1)
    assert hits
    assert "citations" in hits[0].chunk.text


def test_markdown_heading_chunk_strategy_preserves_heading_metadata(tmp_path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "# Setup\n\nInstall dependencies first.\n\n# Usage\n\nRun the bot from the gateway.",
        encoding="utf-8",
    )

    service = KnowledgeService(tmp_path, chunk_strategy="section")
    results = service.ingest_files([str(source)])

    assert results[0].status == "ok"
    hits = service.search("gateway", top_k=1)

    assert hits
    assert hits[0].chunk.heading == "Usage"
    assert "Usage" in hits[0].chunk.text


def test_ingest_service_rejects_files_over_safety_limit(tmp_path) -> None:
    source = tmp_path / "large.txt"
    source.write_text("large document content", encoding="utf-8")

    service = KnowledgeService(tmp_path, max_file_bytes=5)
    results = service.ingest_files([str(source)])

    assert results[0].status == "error"
    assert "file too large" in str(results[0].error)


def test_ingest_service_rejects_excessive_chunk_count(tmp_path) -> None:
    source = tmp_path / "many.txt"
    source.write_text("A\n\nB\n\nC", encoding="utf-8")

    service = KnowledgeService(tmp_path, max_chunks_per_file=2)
    results = service.ingest_files([str(source)])

    assert results[0].status == "error"
    assert "too many chunks" in str(results[0].error)
