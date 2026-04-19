"""MinerU integration wrapper."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path

import httpx

from nanobot.knowledge.types import ParsedDocument, ParsedSection


class MinerUParser:
    """MinerU-backed parsing with command and HTTP API modes."""

    def __init__(
        self,
        command: str | None = None,
        *,
        mode: str = "agent",
        base_url: str = "https://mineru.net",
        api_token: str = "",
        model_version: str = "vlm",
        language: str = "ch",
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        page_range: str | None = None,
        timeout_s: int = 300,
        poll_interval_s: int = 3,
    ):
        self.command = command or ""
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token.strip()
        self.model_version = model_version
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.is_ocr = is_ocr
        self.page_range = page_range
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s

    def is_available(self) -> bool:
        if self.mode == "command":
            return bool(self.command.strip())
        if self.mode == "precision":
            return bool(self.api_token)
        return True

    def parse(self, path: Path) -> ParsedDocument | None:
        """Run a configured MinerU command and normalize the result.

        Supported placeholders in the configured command:
        - ``{input}``: input file path
        - ``{output}``: temp output directory

        The command may either:
        - print JSON to stdout, or
        - write JSON/Markdown/TXT files into ``{output}``
        """

        if not self.is_available():
            return None

        if self.mode == "command":
            return self._parse_with_command(path)
        if self.mode == "precision":
            return self._parse_with_precision_api(path)
        return self._parse_with_agent_api(path)

    def _parse_with_command(self, path: Path) -> ParsedDocument | None:
        with tempfile.TemporaryDirectory(prefix="nanobot-mineru-") as temp_dir:
            output_dir = Path(temp_dir)
            command = self.command.format(input=str(path), output=str(output_dir))
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="ignore",
            )
            if completed.returncode != 0:
                return None

            parsed = self._from_stdout(path, completed.stdout)
            if parsed is not None:
                return parsed
            return self._from_output_dir(path, output_dir)

    def _parse_with_agent_api(self, path: Path) -> ParsedDocument | None:
        payload = {
            "file_name": path.name,
            "language": self.language,
            "enable_table": self.enable_table,
            "is_ocr": self.is_ocr,
            "enable_formula": self.enable_formula,
        }
        if self.page_range:
            payload["page_range"] = self.page_range

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/api/v1/agent/parse/file", json=payload)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 0:
                return None
            data = result.get("data") or {}
            task_id = data.get("task_id")
            file_url = data.get("file_url")
            if not task_id or not file_url:
                return None

            with path.open("rb") as handle:
                put_resp = client.put(file_url, content=handle.read())
            if put_resp.status_code not in (200, 201):
                return None

            markdown = self._poll_agent_markdown(client, str(task_id))
            if not markdown:
                return None
            return self._from_markdown_text(path, markdown, parser="mineru-agent")

    def _poll_agent_markdown(self, client: httpx.Client, task_id: str) -> str | None:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            resp = client.get(f"{self.base_url}/api/v1/agent/parse/{task_id}")
            resp.raise_for_status()
            result = resp.json()
            data = result.get("data") or {}
            state = data.get("state")
            if state == "done":
                markdown_url = data.get("markdown_url")
                if not markdown_url:
                    return None
                md_resp = client.get(str(markdown_url))
                md_resp.raise_for_status()
                return md_resp.text
            if state == "failed":
                return None
            time.sleep(self.poll_interval_s)
        return None

    def _parse_with_precision_api(self, path: Path) -> ParsedDocument | None:
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        payload = {
            "files": [{
                "name": path.name,
                "data_id": path.stem,
                "is_ocr": self.is_ocr,
                **({"page_ranges": self.page_range} if self.page_range else {}),
            }],
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
        }

        with httpx.Client(timeout=self.timeout_s, headers=headers) as client:
            resp = client.post(f"{self.base_url}/api/v4/file-urls/batch", json=payload)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 0:
                return None
            data = result.get("data") or {}
            batch_id = data.get("batch_id")
            file_urls = data.get("file_urls") or []
            if not batch_id or not file_urls:
                return None

            with path.open("rb") as handle, httpx.Client(timeout=self.timeout_s) as upload_client:
                put_resp = upload_client.put(str(file_urls[0]), content=handle.read())
            if put_resp.status_code not in (200, 201):
                return None

            archive_bytes = self._poll_precision_zip(client, str(batch_id), path.name)
            if archive_bytes is None:
                return None
            return self._from_zip_bytes(path, archive_bytes)

    def _poll_precision_zip(self, client: httpx.Client, batch_id: str, file_name: str) -> bytes | None:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            resp = client.get(f"{self.base_url}/api/v4/extract-results/batch/{batch_id}")
            resp.raise_for_status()
            result = resp.json()
            data = result.get("data") or {}
            items = data.get("extract_result") or []
            for item in items:
                if item.get("file_name") != file_name:
                    continue
                state = str(item.get("state", "")).lower()
                status = str(item.get("status", "")).lower()
                is_done = state in {"done", "success", "completed"} or status in {"done", "success", "completed"}
                is_failed = state in {"failed", "error"} or status in {"failed", "error"}
                zip_url = (
                    item.get("full_zip_url")
                    or item.get("full_zip_uri")
                    or item.get("result_zip_url")
                    or item.get("zip_url")
                )
                if not zip_url:
                    result_urls = item.get("result_urls") or item.get("extract_result") or {}
                    if isinstance(result_urls, dict):
                        zip_url = (
                            result_urls.get("full_zip_url")
                            or result_urls.get("full_zip_uri")
                            or result_urls.get("zip_url")
                        )

                if is_done or zip_url:
                    if not zip_url:
                        return None
                    zip_resp = client.get(str(zip_url))
                    zip_resp.raise_for_status()
                    return zip_resp.content
                if is_failed:
                    return None
            time.sleep(self.poll_interval_s)
        return None

    def _from_stdout(self, path: Path, stdout: str) -> ParsedDocument | None:
        content = stdout.strip()
        if not content:
            return None
        if content.startswith("{") or content.startswith("["):
            try:
                return self._from_json_payload(path, json.loads(content))
            except json.JSONDecodeError:
                return None
        return ParsedDocument(
            source_file=path.name,
            file_type="pdf",
            title=path.stem,
            sections=[ParsedSection(text=content)],
            parser="mineru-stdout",
        )

    def _from_zip_bytes(self, path: Path, archive_bytes: bytes) -> ParsedDocument | None:
        try:
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                for name in archive.namelist():
                    if name.endswith("full.md"):
                        markdown = archive.read(name).decode("utf-8", errors="ignore")
                        return self._from_markdown_text(path, markdown, parser="mineru-precision")
                    if name.endswith(".json"):
                        try:
                            payload = json.loads(archive.read(name).decode("utf-8", errors="ignore"))
                        except Exception:
                            continue
                        parsed = self._from_json_payload(path, payload)
                        if parsed is not None:
                            parsed.parser = "mineru-precision"
                            return parsed
        except zipfile.BadZipFile:
            return None
        return None

    def _from_markdown_text(self, path: Path, markdown: str, *, parser: str) -> ParsedDocument | None:
        text = markdown.strip()
        if not text:
            return None

        sections: list[ParsedSection] = []
        current_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if buffer:
                sections.append(ParsedSection(text="\n".join(buffer).strip(), heading=current_heading))
                buffer = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush()
                current_heading = stripped.lstrip("#").strip() or None
                continue
            if stripped:
                buffer.append(stripped)
            elif buffer:
                buffer.append("")
        flush()

        if not sections:
            sections = [ParsedSection(text=text)]
        return ParsedDocument(
            source_file=path.name,
            file_type="pdf",
            title=path.stem,
            sections=sections,
            parser=parser,
        )

    def _from_output_dir(self, path: Path, output_dir: Path) -> ParsedDocument | None:
        json_files = sorted(output_dir.rglob("*.json"))
        for candidate in json_files:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            parsed = self._from_json_payload(path, payload)
            if parsed is not None:
                return parsed

        text_files = list(output_dir.rglob("*.md")) + list(output_dir.rglob("*.txt"))
        for candidate in sorted(text_files):
            text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return ParsedDocument(
                    source_file=path.name,
                    file_type="pdf",
                    title=path.stem,
                    sections=[ParsedSection(text=text)],
                    parser="mineru-text",
                )
        return None

    def _from_json_payload(self, path: Path, payload: object) -> ParsedDocument | None:
        sections: list[ParsedSection] = []

        if isinstance(payload, dict):
            raw_sections = payload.get("sections")
            if isinstance(raw_sections, list):
                for item in raw_sections:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    page = item.get("page")
                    if not isinstance(page, int):
                        page = None
                    heading = item.get("heading")
                    if heading is not None:
                        heading = str(heading).strip() or None
                    sections.append(ParsedSection(text=text, page=page, heading=heading))

            if not sections:
                for key in ("text", "markdown", "content"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        sections.append(ParsedSection(text=value.strip()))
                        break

            if sections:
                title = payload.get("title")
                title_str = str(title).strip() if title is not None else path.stem
                return ParsedDocument(
                    source_file=path.name,
                    file_type="pdf",
                    title=title_str or path.stem,
                    sections=sections,
                    parser="mineru-json",
                )

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if text:
                    page = item.get("page")
                    sections.append(ParsedSection(
                        text=text,
                        page=page if isinstance(page, int) else None,
                        heading=str(item.get("heading")).strip() if item.get("heading") else None,
                    ))
            if sections:
                return ParsedDocument(
                    source_file=path.name,
                    file_type="pdf",
                    title=path.stem,
                    sections=sections,
                    parser="mineru-json",
                )

        return None
