import json
import shutil
import time
from pathlib import Path

from nanobot.knowledge.eval import RetrievalEvalCase, evaluate_retrieval, result_to_dict
from nanobot.knowledge.service import KnowledgeService


ROOT = Path.cwd()
QA_PATH = ROOT / "datasets" / "uda-benchmark" / "extended_qa_info" / "paper_text_qa.json"
OUT = ROOT / "benchmark_results" / "uda_paper_text_evidence_subset"
DOCS = OUT / "docs"
WORKSPACES = OUT / "workspaces"
REPORT_JSON = OUT / "results.json"
REPORT_MD = OUT / "report.md"

MAX_DOCS = 150
MAX_CASES = 300
K_VALUES = [1, 3, 5, 10]


def evidence_to_text(item: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for evidence in item.get("evidence", []) or []:
        if not isinstance(evidence, dict):
            continue
        raw = str(evidence.get("raw_evidence") or "").strip()
        highlighted = str(evidence.get("highlighted_evidence") or "").strip()
        if raw:
            texts.append(raw)
        if highlighted and highlighted != raw:
            texts.append(highlighted)
    return texts


def prepare_dataset() -> tuple[list[str], list[RetrievalEvalCase]]:
    if OUT.exists():
        shutil.rmtree(OUT)
    DOCS.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)

    qa_by_doc = json.loads(QA_PATH.read_text(encoding="utf-8"))
    doc_paths: list[str] = []
    cases: list[RetrievalEvalCase] = []

    for doc_index, doc_id in enumerate(sorted(qa_by_doc), start=1):
        if len(doc_paths) >= MAX_DOCS:
            break
        items = qa_by_doc[doc_id]
        evidence_texts: list[str] = []
        seen: set[str] = set()
        for item in items:
            for text in evidence_to_text(item):
                if text not in seen:
                    evidence_texts.append(text)
                    seen.add(text)
        if not evidence_texts:
            continue
        filename = f"uda_paper_{doc_id}.md"
        path = DOCS / filename
        body = [f"# Paper {doc_id}", ""]
        for index, text in enumerate(evidence_texts, start=1):
            body.append(f"## Evidence {index}")
            body.append(text)
            body.append("")
        path.write_text("\n".join(body), encoding="utf-8")
        doc_paths.append(str(path))

        for item in items:
            question = str(item.get("question") or "").strip()
            if question and len(cases) < MAX_CASES:
                cases.append(RetrievalEvalCase(query=question, relevant=[filename]))

    return doc_paths, cases


def run_experiment(exp: dict[str, object], doc_paths: list[str], cases: list[RetrievalEvalCase]) -> dict[str, object]:
    name = str(exp["name"])
    print(f"\n=== {name} ===", flush=True)
    ws = WORKSPACES / name
    ws.mkdir(parents=True, exist_ok=True)

    init_start = time.perf_counter()
    service = KnowledgeService(
        ws,
        max_file_bytes=5 * 1024 * 1024,
        max_chunks_per_file=200,
        max_chunk_chars=1000,
        chunk_overlap=100,
        chunk_strategy="section",
        embedding_provider=str(exp["embedding_provider"]),
        embedding_model=str(exp.get("embedding_model", "")),
        embedding_dim=int(exp["embedding_dim"]),
        embedding_batch_size=int(exp.get("embedding_batch_size", 64)),
        retrieval_mode=str(exp["retrieval_mode"]),
        reranker_provider=str(exp.get("reranker_provider", "none")),
        reranker_model=str(exp.get("reranker_model", "")),
        reranker_top_k=int(exp.get("reranker_top_k", 20)),
    )
    init_s = time.perf_counter() - init_start

    ingest_start = time.perf_counter()
    ingest_results = service.ingest_files(doc_paths)
    ingest_s = time.perf_counter() - ingest_start
    ok = sum(1 for result in ingest_results if result.status == "ok")
    errors = [result.error for result in ingest_results if result.status != "ok"]
    if errors:
        print(f"ingest errors: {errors[:3]}", flush=True)

    eval_start = time.perf_counter()
    metric = evaluate_retrieval(
        service,
        cases,
        k_values=K_VALUES,
        retrieval_mode=str(exp["retrieval_mode"]),
    )
    eval_s = time.perf_counter() - eval_start

    row = {
        "name": name,
        "retrieval_mode": exp["retrieval_mode"],
        "embedding_provider": exp["embedding_provider"],
        "reranker_provider": exp.get("reranker_provider", "none"),
        "ingested_docs": ok,
        "init_seconds": round(init_s, 3),
        "ingest_seconds": round(ingest_s, 3),
        "eval_seconds": round(eval_s, 3),
        "avg_query_ms": round(eval_s * 1000 / max(1, len(cases)), 3),
        "metrics": result_to_dict(metric),
    }
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return row


def write_report(results: dict[str, object]) -> None:
    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# UDA-Benchmark paper text evidence retrieval benchmark",
        "",
        "- Dataset: UDA-Benchmark paper_text QA subset",
        f"- Documents: {results['docs']} papers represented by gold evidence snippets",
        f"- Queries: {results['cases']} paper questions",
        "- Relevant label: original paper file should appear in top-k retrieval results",
        "- Note: this measures retrieval on paper evidence text, not end-to-end PDF parsing.",
        "",
        "| Experiment | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg query ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["experiments"]:
        metrics = row["metrics"]
        rec = metrics["recall_at_k"]
        lines.append(
            f"| {row['name']} | {rec.get('1', 0):.4f} | {rec.get('3', 0):.4f} | "
            f"{rec.get('5', 0):.4f} | {rec.get('10', 0):.4f} | {metrics['mrr']:.4f} | "
            f"{row['avg_query_ms']:.2f} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_JSON}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)


def main() -> None:
    doc_paths, cases = prepare_dataset()
    experiments = [
        {
            "name": "baseline_hashing_hybrid",
            "embedding_provider": "hashing",
            "embedding_model": "",
            "embedding_dim": 384,
            "retrieval_mode": "hybrid",
            "reranker_provider": "none",
            "reranker_model": "",
        },
        {
            "name": "ours_bge_vector",
            "embedding_provider": "bge-m3",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "embedding_batch_size": 8,
            "retrieval_mode": "vector",
            "reranker_provider": "none",
            "reranker_model": "",
        },
        {
            "name": "ours_bge_hybrid",
            "embedding_provider": "bge-m3",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "embedding_batch_size": 8,
            "retrieval_mode": "hybrid",
            "reranker_provider": "none",
            "reranker_model": "",
        },
        {
            "name": "ours_bge_hybrid_rerank",
            "embedding_provider": "bge-m3",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "embedding_batch_size": 8,
            "retrieval_mode": "hybrid",
            "reranker_provider": "bge-reranker",
            "reranker_model": "BAAI/bge-reranker-base",
            "reranker_top_k": 20,
        },
    ]
    results: dict[str, object] = {
        "dataset": "UDA-Benchmark paper_text evidence subset",
        "source": str(QA_PATH),
        "docs": len(doc_paths),
        "cases": len(cases),
        "k_values": K_VALUES,
        "experiments": [],
    }
    for exp in experiments:
        results["experiments"].append(run_experiment(exp, doc_paths, cases))
    write_report(results)


if __name__ == "__main__":
    main()
