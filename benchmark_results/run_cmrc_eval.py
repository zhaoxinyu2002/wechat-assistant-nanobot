import json
import shutil
import time
from pathlib import Path

from nanobot.knowledge.eval import RetrievalEvalCase, evaluate_retrieval, result_to_dict
from nanobot.knowledge.service import KnowledgeService


ROOT = Path.cwd()
DATA = ROOT / "datasets" / "cmrc2018" / "cmrc2018_dev.json"
OUT = ROOT / "benchmark_results" / "cmrc2018_dev_subset"
DOCS = OUT / "docs"
WORKSPACES = OUT / "workspaces"
REPORT_JSON = OUT / "results.json"
REPORT_MD = OUT / "report.md"

MAX_DOCS = 150
MAX_CASES = 300
K_VALUES = [1, 3, 5, 10]


def prepare_dataset() -> tuple[list[str], list[RetrievalEvalCase]]:
    if OUT.exists():
        shutil.rmtree(OUT)
    DOCS.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)

    raw = json.loads(DATA.read_text(encoding="utf-8"))
    doc_paths: list[str] = []
    cases: list[RetrievalEvalCase] = []
    doc_count = 0

    for article in raw["data"]:
        for para in article.get("paragraphs", []):
            if doc_count >= MAX_DOCS:
                break
            context = str(para.get("context") or "").strip()
            if not context:
                continue
            filename = f"cmrc_dev_p{doc_count:04d}.txt"
            path = DOCS / filename
            path.write_text(context, encoding="utf-8")
            doc_paths.append(str(path))
            for qa in para.get("qas", []):
                question = str(qa.get("question") or "").strip()
                if question and len(cases) < MAX_CASES:
                    cases.append(RetrievalEvalCase(query=question, relevant=[filename]))
            doc_count += 1
        if doc_count >= MAX_DOCS:
            break
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
        max_chunks_per_file=100,
        max_chunk_chars=1200,
        chunk_overlap=120,
        chunk_strategy="recursive",
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
        "# CMRC2018 dev subset retrieval benchmark",
        "",
        "- Dataset: CMRC2018 dev subset",
        f"- Documents: {results['docs']} paragraphs as .txt files",
        f"- Queries: {results['cases']} questions",
        "- Relevant label: original paragraph file should appear in top-k retrieval results",
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
            "name": "baseline_hashing_keyword",
            "embedding_provider": "hashing",
            "embedding_model": "",
            "embedding_dim": 384,
            "retrieval_mode": "keyword",
            "reranker_provider": "none",
            "reranker_model": "",
        },
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
        "dataset": "CMRC2018 dev subset",
        "source": str(DATA),
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
