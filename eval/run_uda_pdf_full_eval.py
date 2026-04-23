import argparse
import json
import os
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
OUT = ROOT / "benchmark_results" / "uda_pdf_full"
PDF_DIR = OUT / "pdfs"
WORKSPACES = OUT / "workspaces"
STATUS_DIR = OUT / "status"
RESULTS_JSON = OUT / "results.json"
REPORT_MD = OUT / "report.md"

K_VALUES = [1, 3, 5]

# Prefer cached local model files for long-running eval jobs. This avoids
# brittle metadata HEAD requests when the environment blocks outbound access.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def readable_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}+-=*/%_<>#'\"")
    return readable / len(text)


def load_pdf_rows(
    limit: int | None = None,
    *,
    max_bytes: int | None = None,
    min_questions: int = 1,
    offset: int = 0,
) -> list[dict[str, object]]:
    qa_by_doc = json.loads(QA_PATH.read_text(encoding="utf-8"))
    with zipfile.ZipFile(PDF_ZIP) as archive:
        names = {Path(name).stem: name for name in archive.namelist() if name.lower().endswith(".pdf")}
        rows = []
        for doc_id in qa_by_doc:
            if doc_id not in names:
                continue
            info = archive.getinfo(names[doc_id])
            if max_bytes is not None and info.file_size > max_bytes:
                continue
            if len(qa_by_doc[doc_id]) < min_questions:
                continue
            rows.append({
                "doc_id": doc_id,
                "member": names[doc_id],
                "size": info.file_size,
                "questions": len(qa_by_doc[doc_id]),
            })
    rows.sort(key=lambda row: (int(row["size"]), str(row["doc_id"])))
    if offset > 0:
        rows = rows[offset:]
    return rows[:limit] if limit else rows


def extract_pdf(row: dict[str, object]) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    target = PDF_DIR / f"{row['doc_id']}.pdf"
    if target.exists() and target.stat().st_size == int(row["size"]):
        return target
    with zipfile.ZipFile(PDF_ZIP) as archive:
        target.write_bytes(archive.read(str(row["member"])))
    return target


def build_cases(rows: list[dict[str, object]]) -> list[RetrievalEvalCase]:
    qa_by_doc = json.loads(QA_PATH.read_text(encoding="utf-8"))
    allowed = {str(row["doc_id"]) for row in rows}
    cases: list[RetrievalEvalCase] = []
    for doc_id in sorted(allowed):
        for item in qa_by_doc.get(doc_id, []):
            question = str(item.get("question") or "").strip()
            if question:
                cases.append(RetrievalEvalCase(query=question, relevant=[f"{doc_id}.pdf"]))
    return cases


def status_path(experiment: str) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    return STATUS_DIR / f"{experiment}.jsonl"


def load_status(experiment: str) -> dict[str, dict[str, object]]:
    path = status_path(experiment)
    if not path.exists():
        return {}
    statuses: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statuses[str(row["doc_id"])] = row
    return statuses


def append_status(experiment: str, row: dict[str, object]) -> None:
    path = status_path(experiment)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_service(workspace: Path, *, parser_pdf: str) -> KnowledgeService:
    cfg = load_config().knowledge
    return KnowledgeService(
        workspace,
        max_file_bytes=20 * 1024 * 1024,
        max_chunks_per_file=1000,
        max_chunk_chars=1200,
        chunk_overlap=150,
        chunk_strategy="section",
        embedding_provider="bge-m3",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        embedding_batch_size=8,
        retrieval_mode="vector",
        reranker_provider="none",
        parser_pdf=parser_pdf,
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
        mineru_timeout_s=1200,
        mineru_poll_interval_s=5,
    )


def ingest_experiment(
    experiment: str,
    rows: list[dict[str, object]],
    *,
    parser_pdf: str,
    cooldown_s: float = 0.0,
) -> None:
    workspace = WORKSPACES / experiment
    workspace.mkdir(parents=True, exist_ok=True)
    service = make_service(workspace, parser_pdf=parser_pdf)
    statuses = load_status(experiment)
    done = sum(1 for row in statuses.values() if row.get("status") == "ok")
    print(f"{experiment}: already ok {done}/{len(rows)}", flush=True)

    for index, row in enumerate(rows, start=1):
        doc_id = str(row["doc_id"])
        existing = statuses.get(doc_id)
        if existing and existing.get("status") == "ok":
            continue

        pdf_path = extract_pdf(row)
        start = time.perf_counter()
        result = service.ingest_files([str(pdf_path)])[0]
        elapsed = time.perf_counter() - start
        status_row = {
            "doc_id": doc_id,
            "size": row["size"],
            "questions": row["questions"],
            "status": result.status,
            "chunks_created": result.chunks_created,
            "parser": result.parser,
            "error": result.error,
            "seconds": round(elapsed, 3),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_status(experiment, status_row)
        statuses[doc_id] = status_row
        ok = sum(1 for item in statuses.values() if item.get("status") == "ok")
        print(
            f"{experiment}: {index}/{len(rows)} {doc_id} {result.status} "
            f"chunks={result.chunks_created} elapsed={elapsed:.1f}s ok={ok}",
            flush=True,
        )
        if cooldown_s > 0:
            time.sleep(cooldown_s)


def collect_parse_stats(experiment: str, rows: list[dict[str, object]]) -> dict[str, object]:
    workspace = WORKSPACES / experiment
    parsed_dir = workspace / "knowledge" / "parsed"
    chunks_dir = workspace / "knowledge" / "chunks"
    total_chars = 0
    total_chunks = 0
    ratios = []
    parsed_files = 0
    for row in rows:
        doc_id = str(row["doc_id"])
        parsed_path = parsed_dir / f"{doc_id}.json"
        chunks_path = chunks_dir / f"{doc_id}.jsonl"
        if parsed_path.exists():
            payload = json.loads(parsed_path.read_text(encoding="utf-8"))
            text = normalize_text("\n".join(str(section.get("text") or "") for section in payload.get("sections", []) if isinstance(section, dict)))
            total_chars += len(text)
            ratios.append(readable_ratio(text))
            parsed_files += 1
        if chunks_path.exists():
            total_chunks += sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "parsed_files": parsed_files,
        "total_text_chars": total_chars,
        "total_chunks": total_chunks,
        "avg_chunks_per_file": total_chunks / max(1, len(rows)),
        "avg_readable_ratio": sum(ratios) / max(1, len(ratios)),
    }


def evaluate_experiment(experiment: str, rows: list[dict[str, object]], cases: list[RetrievalEvalCase]) -> dict[str, object]:
    workspace = WORKSPACES / experiment
    service = make_service(workspace, parser_pdf="basic")
    start = time.perf_counter()
    metrics = evaluate_retrieval(service, cases, k_values=K_VALUES, retrieval_mode="vector")
    eval_s = time.perf_counter() - start
    return {
        "name": experiment,
        "docs": len(rows),
        "cases": len(cases),
        "eval_seconds": round(eval_s, 3),
        "avg_query_ms": round(eval_s * 1000 / max(1, len(cases)), 3),
        "parse_stats": collect_parse_stats(experiment, rows),
        "metrics": result_to_dict(metrics),
    }


def write_report(results: dict[str, object]) -> None:
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# UDA PDF Full Benchmark",
        "",
        f"- PDFs: {results['docs']}",
        f"- Questions: {results['cases']}",
        "- Scope: all UDA paper_text PDFs with matching QA labels",
        "- MinerU setting: first 6 pages, OCR disabled, BGE-M3 vector retrieval",
        "",
        "| Experiment | Parsed files | Text chars | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Avg query ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["experiments"]:
        stats = row["parse_stats"]
        metrics = row["metrics"]
        rec = metrics["recall_at_k"]
        lines.append(
            f"| {row['name']} | {stats['parsed_files']}/{results['docs']} | "
            f"{stats['total_text_chars']} | {stats['total_chunks']} | "
            f"{rec.get('1', 0):.4f} | {rec.get('3', 0):.4f} | {rec.get('5', 0):.4f} | "
            f"{metrics['mrr']:.4f} | {row['avg_query_ms']:.2f} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_all(rows: list[dict[str, object]]) -> None:
    cases = build_cases(rows)
    experiments = []
    for experiment in ["without_mineru_basic_fallback", "with_mineru"]:
        if (WORKSPACES / experiment / "knowledge" / "index" / "knowledge.db").exists():
            experiments.append(evaluate_experiment(experiment, rows, cases))
    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_zip": str(PDF_ZIP),
        "qa_path": str(QA_PATH),
        "docs": len(rows),
        "cases": len(cases),
        "k_values": K_VALUES,
        "experiments": experiments,
    }
    write_report(results)
    print(f"Wrote {RESULTS_JSON}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["basic", "mineru", "both", "eval"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--min-questions", type=int, default=1)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--out-name", default="uda_pdf_full")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    global OUT, PDF_DIR, WORKSPACES, STATUS_DIR, RESULTS_JSON, REPORT_MD
    OUT = ROOT / "benchmark_results" / args.out_name
    PDF_DIR = OUT / "pdfs"
    WORKSPACES = OUT / "workspaces"
    STATUS_DIR = OUT / "status"
    RESULTS_JSON = OUT / "results.json"
    REPORT_MD = OUT / "report.md"

    rows = load_pdf_rows(
        limit=args.limit,
        max_bytes=args.max_bytes,
        min_questions=args.min_questions,
        offset=args.offset,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selected PDFs: {len(rows)}; total MB={sum(int(r['size']) for r in rows)/1024/1024:.2f}", flush=True)

    if args.fresh:
        # Do not remove OUT itself: on Windows the redirected run.log/run.err.log
        # files may already be open by the parent process.
        for target in (PDF_DIR, WORKSPACES, STATUS_DIR, RESULTS_JSON, REPORT_MD):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        OUT.mkdir(parents=True, exist_ok=True)
        WORKSPACES.mkdir(parents=True, exist_ok=True)
        STATUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"basic", "both"}:
        ingest_experiment(
            "without_mineru_basic_fallback",
            rows,
            parser_pdf="basic",
            cooldown_s=args.cooldown_s,
        )
    if args.stage in {"mineru", "both"}:
        ingest_experiment(
            "with_mineru",
            rows,
            parser_pdf="mineru",
            cooldown_s=args.cooldown_s,
        )
    if args.stage == "eval":
        evaluate_all(rows)


if __name__ == "__main__":
    main()
