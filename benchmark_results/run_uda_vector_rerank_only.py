import json
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
MAX_CASES = 300
K_VALUES = [1, 3, 5, 10]


def build_cases() -> list[RetrievalEvalCase]:
    qa_by_doc = json.loads(QA_PATH.read_text(encoding="utf-8"))
    available = {path.name for path in DOCS.glob("uda_paper_*.md")}
    cases: list[RetrievalEvalCase] = []
    for doc_id in sorted(qa_by_doc):
        filename = f"uda_paper_{doc_id}.md"
        if filename not in available:
            continue
        for item in qa_by_doc[doc_id]:
            question = str(item.get("question") or "").strip()
            if question and len(cases) < MAX_CASES:
                cases.append(RetrievalEvalCase(query=question, relevant=[filename]))
    return cases


def main() -> None:
    doc_paths = [str(path) for path in sorted(DOCS.glob("uda_paper_*.md"))]
    cases = build_cases()
    name = "ours_bge_vector_rerank"
    ws = WORKSPACES / name
    ws.mkdir(parents=True, exist_ok=True)

    print(f"=== {name} ===", flush=True)
    init_start = time.perf_counter()
    service = KnowledgeService(
        ws,
        max_file_bytes=5 * 1024 * 1024,
        max_chunks_per_file=200,
        max_chunk_chars=1000,
        chunk_overlap=100,
        chunk_strategy="section",
        embedding_provider="bge-m3",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        embedding_batch_size=8,
        retrieval_mode="vector",
        reranker_provider="bge-reranker",
        reranker_model="BAAI/bge-reranker-base",
        reranker_top_k=20,
    )
    init_s = time.perf_counter() - init_start

    ingest_start = time.perf_counter()
    ingest_results = service.ingest_files(doc_paths)
    ingest_s = time.perf_counter() - ingest_start
    ok = sum(1 for result in ingest_results if result.status == "ok")

    eval_start = time.perf_counter()
    metric = evaluate_retrieval(service, cases, k_values=K_VALUES, retrieval_mode="vector")
    eval_s = time.perf_counter() - eval_start
    row = {
        "name": name,
        "retrieval_mode": "vector",
        "embedding_provider": "bge-m3",
        "reranker_provider": "bge-reranker",
        "ingested_docs": ok,
        "init_seconds": round(init_s, 3),
        "ingest_seconds": round(ingest_s, 3),
        "eval_seconds": round(eval_s, 3),
        "avg_query_ms": round(eval_s * 1000 / max(1, len(cases)), 3),
        "metrics": result_to_dict(metric),
    }
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    results = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    results["experiments"] = [
        item for item in results["experiments"] if item["name"] != name
    ]
    results["experiments"].append(row)
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
    for item in results["experiments"]:
        metrics = item["metrics"]
        rec = metrics["recall_at_k"]
        lines.append(
            f"| {item['name']} | {rec.get('1', 0):.4f} | {rec.get('3', 0):.4f} | "
            f"{rec.get('5', 0):.4f} | {rec.get('10', 0):.4f} | {metrics['mrr']:.4f} | "
            f"{item['avg_query_ms']:.2f} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_JSON}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)


if __name__ == "__main__":
    main()
