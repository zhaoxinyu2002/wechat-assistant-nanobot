import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.knowledge.eval import RetrievalEvalCase, evaluate_retrieval, result_to_dict
from nanobot.knowledge.service import KnowledgeService


ROOT = Path.cwd()
QA_PATH = ROOT / "datasets" / "uda-benchmark" / "extended_qa_info" / "paper_text_qa.json"
PDF_ZIP = ROOT / "datasets" / "uda-benchmark" / "src_doc_files" / "paper_docs.zip"
OUT = ROOT / "benchmark_results" / "uda_pdf_mineru_subset"
PDF_DIR = OUT / "pdfs"
WORKSPACES = OUT / "workspaces"
REPORT_JSON = OUT / "results.json"
REPORT_MD = OUT / "report.md"

# Small, low-risk PDFs from UDA-Benchmark. They are intentionally tiny so the
# benchmark exercises the PDF path without stressing the laptop.
DOC_IDS = [
    "1805.10824",
    "1909.11467",
    "1901.10133",
    "1608.01884",
    "1901.04899",
]
K_VALUES = [1, 3, 5]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}+-=*/%_<>#'\"")
    return readable / len(text)


def load_pdf_map() -> dict[str, str]:
    with zipfile.ZipFile(PDF_ZIP) as archive:
        return {
            Path(name).stem: name
            for name in archive.namelist()
            if name.lower().endswith(".pdf")
        }


def extract_pdfs() -> list[str]:
    if OUT.exists():
        shutil.rmtree(OUT)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)

    pdf_map = load_pdf_map()
    paths: list[str] = []
    with zipfile.ZipFile(PDF_ZIP) as archive:
        for doc_id in DOC_IDS:
            member = pdf_map[doc_id]
            target = PDF_DIR / f"{doc_id}.pdf"
            target.write_bytes(archive.read(member))
            paths.append(str(target))
    return paths


def build_cases() -> list[RetrievalEvalCase]:
    qa_by_doc = json.loads(QA_PATH.read_text(encoding="utf-8"))
    cases: list[RetrievalEvalCase] = []
    for doc_id in DOC_IDS:
        filename = f"{doc_id}.pdf"
        for item in qa_by_doc.get(doc_id, []):
            question = str(item.get("question") or "").strip()
            if question:
                cases.append(RetrievalEvalCase(query=question, relevant=[filename]))
    return cases


def collect_parse_stats(workspace: Path, source_files: list[str]) -> dict[str, object]:
    parsed_dir = workspace / "knowledge" / "parsed"
    chunks_dir = workspace / "knowledge" / "chunks"
    per_file: list[dict[str, object]] = []
    total_text_chars = 0
    total_chunks = 0
    ok_files = 0

    for source in source_files:
        stem = Path(source).stem
        parsed_path = parsed_dir / f"{stem}.json"
        chunks_path = chunks_dir / f"{stem}.jsonl"
        item: dict[str, object] = {
            "source_file": Path(source).name,
            "parsed": False,
            "parser": None,
            "sections": 0,
            "chunks": 0,
            "text_chars": 0,
            "readable_ratio": 0.0,
        }
        if parsed_path.exists():
            payload = json.loads(parsed_path.read_text(encoding="utf-8"))
            sections = payload.get("sections") or []
            text = "\n".join(str(section.get("text") or "") for section in sections if isinstance(section, dict))
            text = normalize_text(text)
            item.update({
                "parsed": True,
                "parser": payload.get("parser"),
                "sections": len(sections),
                "text_chars": len(text),
                "readable_ratio": round(readable_ratio(text), 4),
            })
            total_text_chars += len(text)
            ok_files += 1
        if chunks_path.exists():
            lines = [line for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            item["chunks"] = len(lines)
            total_chunks += len(lines)
        per_file.append(item)

    return {
        "parsed_files": ok_files,
        "parse_success_rate": ok_files / max(1, len(source_files)),
        "total_text_chars": total_text_chars,
        "total_chunks": total_chunks,
        "avg_text_chars_per_file": total_text_chars / max(1, len(source_files)),
        "avg_chunks_per_file": total_chunks / max(1, len(source_files)),
        "files": per_file,
    }


def service_for(exp: dict[str, object], workspace: Path) -> KnowledgeService:
    cfg = load_config().knowledge
    return KnowledgeService(
        workspace,
        max_file_bytes=3 * 1024 * 1024,
        max_chunks_per_file=300,
        max_chunk_chars=1200,
        chunk_overlap=150,
        chunk_strategy="section",
        embedding_provider="bge-m3",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        embedding_batch_size=8,
        retrieval_mode="vector",
        reranker_provider="none",
        parser_pdf=str(exp["parser_pdf"]),
        mineru_command=cfg.mineru_command,
        mineru_mode=cfg.mineru_mode,
        mineru_base_url=cfg.mineru_base_url,
        mineru_api_token=cfg.mineru_api_token,
        mineru_model_version=cfg.mineru_model_version,
        mineru_language="en",
        mineru_enable_table=cfg.mineru_enable_table,
        mineru_enable_formula=cfg.mineru_enable_formula,
        mineru_is_ocr=False,
        mineru_page_range="1-6",
        mineru_timeout_s=900,
        mineru_poll_interval_s=5,
    )


def run_experiment(exp: dict[str, object], pdf_paths: list[str], cases: list[RetrievalEvalCase]) -> dict[str, object]:
    name = str(exp["name"])
    print(f"\n=== {name} ===", flush=True)
    workspace = WORKSPACES / name
    workspace.mkdir(parents=True, exist_ok=True)

    init_start = time.perf_counter()
    service = service_for(exp, workspace)
    init_s = time.perf_counter() - init_start

    ingest_start = time.perf_counter()
    ingest_results = service.ingest_files(pdf_paths)
    ingest_s = time.perf_counter() - ingest_start
    errors = [
        {"path": result.path, "error": result.error}
        for result in ingest_results
        if result.status != "ok"
    ]
    ok = sum(1 for result in ingest_results if result.status == "ok")
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2), flush=True)

    eval_start = time.perf_counter()
    metric = evaluate_retrieval(service, cases, k_values=K_VALUES, retrieval_mode="vector")
    eval_s = time.perf_counter() - eval_start

    row = {
        "name": name,
        "parser_pdf": exp["parser_pdf"],
        "ingested_docs": ok,
        "ingest_errors": errors,
        "init_seconds": round(init_s, 3),
        "ingest_seconds": round(ingest_s, 3),
        "eval_seconds": round(eval_s, 3),
        "avg_query_ms": round(eval_s * 1000 / max(1, len(cases)), 3),
        "parse_stats": collect_parse_stats(workspace, pdf_paths),
        "metrics": result_to_dict(metric),
    }
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return row


def write_report(results: dict[str, object]) -> None:
    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# UDA-Benchmark PDF MinerU parsing benchmark",
        "",
        "- Dataset: UDA-Benchmark paper_text QA with real PDFs from paper_docs.zip",
        f"- PDFs: {results['docs']} small papers",
        f"- Queries: {results['cases']} paper questions",
        "- Retrieval: BGE-M3 vector search",
        "- Label: original PDF should appear in top-k retrieval results",
        "- Scope: first 6 pages per PDF for a safe small-sample test",
        "",
        "| Experiment | Parsed files | Text chars | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Ingest s | Avg query ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["experiments"]:
        stats = row["parse_stats"]
        metrics = row["metrics"]
        rec = metrics["recall_at_k"]
        lines.append(
            f"| {row['name']} | {stats['parsed_files']}/{results['docs']} | "
            f"{stats['total_text_chars']} | {stats['total_chunks']} | "
            f"{rec.get('1', 0):.4f} | {rec.get('3', 0):.4f} | {rec.get('5', 0):.4f} | "
            f"{metrics['mrr']:.4f} | {row['ingest_seconds']:.2f} | {row['avg_query_ms']:.2f} |"
        )

    lines.append("")
    lines.append("## Per-file parse stats")
    lines.append("")
    lines.append("| Experiment | File | Parser | Sections | Chunks | Text chars | Readable ratio |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in results["experiments"]:
        for item in row["parse_stats"]["files"]:
            lines.append(
                f"| {row['name']} | {item['source_file']} | {item['parser']} | "
                f"{item['sections']} | {item['chunks']} | {item['text_chars']} | "
                f"{item['readable_ratio']:.4f} |"
            )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_JSON}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)


def main() -> None:
    pdf_paths = extract_pdfs()
    cases = build_cases()
    experiments = [
        {"name": "without_mineru_basic_fallback", "parser_pdf": "basic"},
        {"name": "with_mineru_agent", "parser_pdf": "mineru"},
    ]
    results: dict[str, object] = {
        "dataset": "UDA-Benchmark paper_text PDF subset",
        "pdf_zip": str(PDF_ZIP),
        "qa_path": str(QA_PATH),
        "doc_ids": DOC_IDS,
        "docs": len(pdf_paths),
        "cases": len(cases),
        "k_values": K_VALUES,
        "experiments": [],
    }
    for exp in experiments:
        results["experiments"].append(run_experiment(exp, pdf_paths, cases))
    write_report(results)


if __name__ == "__main__":
    main()
