from nanobot.knowledge.eval import evaluate_retrieval, load_eval_cases
from nanobot.knowledge.service import KnowledgeService


def test_retrieval_eval_reports_recall_hit_rate_and_mrr(tmp_path) -> None:
    source = tmp_path / "guide.txt"
    source.write_text("RAG evaluation measures recall and reciprocal rank.", encoding="utf-8")
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"query":"reciprocal rank","relevant":["reciprocal rank"]}\n',
        encoding="utf-8",
    )

    service = KnowledgeService(tmp_path)
    service.ingest_files([str(source)])

    result = evaluate_retrieval(service, load_eval_cases(dataset), k_values=[1, 3])

    assert result.cases == 1
    assert result.recall_at_k[1] == 1.0
    assert result.hit_rate_at_k[3] == 1.0
    assert result.mrr == 1.0
