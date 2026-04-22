import json
import time
from pathlib import Path

from nanobot.knowledge.service import KnowledgeService


ROOT = Path.cwd()
OUT = ROOT / "benchmark_results" / "uda_pdf_mineru_subset"
WORKSPACES = OUT / "workspaces"
RESULTS_JSON = OUT / "results.json"
REPORT_MD = OUT / "report.md"
K_VALUES = [1, 3, 5]


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def doc_level_metrics(service: KnowledgeService, cases: list[dict[str, object]]) -> dict[str, object]:
    hits_at_k = {k: 0 for k in K_VALUES}
    rr_total = 0.0
    for case in cases:
        query = str(case["query"])
        expected = str(case["source_file"])
        hits = service.search(query, top_k=50, retrieval_mode="vector")
        unique_sources: list[str] = []
        seen = set()
        for hit in hits:
            source = hit.chunk.source_file
            if source in seen:
                continue
            unique_sources.append(source)
            seen.add(source)
            if len(unique_sources) >= max(K_VALUES):
                break
        rank = None
        for idx, source in enumerate(unique_sources, start=1):
            if source == expected:
                rank = idx
                break
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        rr_total += reciprocal_rank(rank)
    total = max(1, len(cases))
    return {
        "cases": len(cases),
        "recall_at_k": {str(k): hits_at_k[k] / total for k in K_VALUES},
        "mrr": rr_total / total,
    }


def build_cases(results: dict[str, object]) -> list[dict[str, object]]:
    qa_path = Path(str(results["qa_path"]))
    qa_by_doc = json.loads(qa_path.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    for doc_id in results["doc_ids"]:
        for item in qa_by_doc.get(doc_id, []):
            question = str(item.get("question") or "").strip()
            if question:
                cases.append({"query": question, "source_file": f"{doc_id}.pdf"})
    return cases


def make_service(workspace: Path) -> KnowledgeService:
    return KnowledgeService(
        workspace,
        embedding_provider="bge-m3",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        embedding_batch_size=8,
        retrieval_mode="vector",
        reranker_provider="none",
        parser_pdf="basic",
    )


def main() -> None:
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    cases = build_cases(results)
    for row in results["experiments"]:
        name = row["name"]
        print(f"=== doc-level {name} ===", flush=True)
        service = make_service(WORKSPACES / name)
        start = time.perf_counter()
        row["doc_level_metrics"] = doc_level_metrics(service, cases)
        row["doc_level_eval_seconds"] = round(time.perf_counter() - start, 3)
        print(json.dumps(row["doc_level_metrics"], ensure_ascii=False, indent=2), flush=True)

    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    existing = REPORT_MD.read_text(encoding="utf-8")
    marker = "\n## Doc-level retrieval metrics\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    lines = [
        existing.rstrip(),
        "",
        "## Doc-level retrieval metrics",
        "",
        "This collapses multiple chunks from the same PDF into one source document before computing Recall@K.",
        "",
        "| Experiment | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["experiments"]:
        metrics = row["doc_level_metrics"]
        rec = metrics["recall_at_k"]
        lines.append(
            f"| {row['name']} | {rec.get('1', 0):.4f} | {rec.get('3', 0):.4f} | "
            f"{rec.get('5', 0):.4f} | {metrics['mrr']:.4f} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_JSON}", flush=True)
    print(f"Wrote {REPORT_MD}", flush=True)


if __name__ == "__main__":
    main()
