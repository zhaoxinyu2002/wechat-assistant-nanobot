"""Document parsing entrypoints for the knowledge base."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from nanobot.knowledge.mineru_parser import MinerUParser
from nanobot.knowledge.types import ParsedDocument, ParsedSection


class DocumentParser:
    """Parse supported document types into a normalized structure."""

    def __init__(self, pdf_parser: str = "mineru", **mineru_kwargs):
        self.pdf_parser = pdf_parser
        self.mineru = MinerUParser(**mineru_kwargs)

    def parse_file(self, path: str | Path) -> ParsedDocument:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            return self._parse_text(file_path, file_type=suffix.lstrip("."))
        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        if suffix == ".docx":
            return self._parse_docx(file_path)

        raise ValueError(f"unsupported format: {suffix or 'unknown'}")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        if self.pdf_parser == "mineru":
            parsed = self.mineru.parse(path)
            if parsed is not None:
                return parsed
        return self._parse_text(path, file_type="pdf", parser="basic-fallback")

    @staticmethod
    def _parse_text(path: Path, *, file_type: str, parser: str = "basic") -> ParsedDocument:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").strip()
        if not text:
            raise ValueError("document has no readable text content")

        sections = [
            ParsedSection(text=part.strip())
            for part in text.split("\n\n")
            if part.strip()
        ]
        if not sections:
            sections = [ParsedSection(text=text)]

        return ParsedDocument(
            source_file=path.name,
            file_type=file_type,
            title=path.stem,
            sections=sections,
            parser=parser,
        )

    @staticmethod
    def _parse_docx(path: Path) -> ParsedDocument:
        try:
            with zipfile.ZipFile(path) as archive:
                document_xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("docx document.xml not found") from exc
        except zipfile.BadZipFile as exc:
            raise ValueError("invalid docx file") from exc

        root = ET.fromstring(document_xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        paragraphs: list[str] = []
        for para in root.findall(".//w:p", ns):
            texts = [node.text or "" for node in para.findall(".//w:t", ns)]
            merged = "".join(texts).strip()
            if merged:
                paragraphs.append(merged)
        if not paragraphs:
            raise ValueError("document has no readable text content")

        sections = [ParsedSection(text=text) for text in paragraphs]
        title = paragraphs[0][:80] if paragraphs else path.stem
        title = re.sub(r"\s+", " ", title).strip() or path.stem
        return ParsedDocument(
            source_file=path.name,
            file_type="docx",
            title=title,
            sections=sections,
            parser="docx-basic",
        )
