"""Retrieval evaluation utilities for the local knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.knowledge.service import KnowledgeService
from nanobot.knowledge.types import SearchHit


@dataclass(slots=True)
class RetrievalEvalCase:
    """One retrieval evaluation query."""

    query: str
    relevant: list[str]
    source_filter: str | None = None


@dataclass(slots=True)
class RetrievalEvalResult:
    """Aggregated retrieval metrics."""

    cases: int
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float


def load_eval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    """Load JSON or JSONL retrieval eval cases.

    Each case should contain:
    - `query`: user query
    - `relevant`: strings matched against chunk_id, source_file, heading, or text
    - optional `source_filter`: restrict retrieval to one file
    """

    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        loaded = json.loads(raw)
        rows = loaded.get("cases", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list):
        raise ValueError("eval dataset must be a JSON array, a {cases: [...]} object, or JSONL")

    cases: list[RetrievalEvalCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"case {index} must be an object")
        query = str(row.get("query") or "").strip()
        relevant = row.get("relevant") or row.get("expected") or []
        if isinstance(relevant, str):
            relevant = [relevant]
        relevant = [str(item).strip() for item in relevant if str(item).strip()]
        if not query or not relevant:
            raise ValueError(f"case {index} requires non-empty query and relevant")
        source_filter = row.get("source_filter") or row.get("source")
        cases.append(RetrievalEvalCase(
            query=query,
            relevant=relevant,
            source_filter=str(source_filter) if source_filter else None,
        ))
    return cases


def evaluate_retrieval(
    service: KnowledgeService,
    cases: list[RetrievalEvalCase],
    *,
    k_values: list[int] | None = None,
    retrieval_mode: str | None = None,
) -> RetrievalEvalResult:
    """Compute Recall@K, HitRate@K, and MRR for retrieval cases."""

    k_values = sorted(set(k_values or [1, 3, 5]))
    if not cases:
        return RetrievalEvalResult(cases=0, recall_at_k={}, hit_rate_at_k={}, mrr=0.0)

    max_k = max(k_values)
    recall_totals = {k: 0.0 for k in k_values}
    hit_totals = {k: 0.0 for k in k_values}
    reciprocal_rank_total = 0.0

    for case in cases:
        hits = service.search(
            case.query,
            top_k=max_k,
            source_filter=case.source_filter,
            retrieval_mode=retrieval_mode,
        )
        match_positions = _match_positions(hits, case.relevant)
        for k in k_values:
            found = [pos for pos in match_positions if pos <= k]
            recall_totals[k] += min(len(found), len(case.relevant)) / len(case.relevant)
            hit_totals[k] += 1.0 if found else 0.0
        if match_positions:
            reciprocal_rank_total += 1.0 / min(match_positions)

    total = float(len(cases))
    return RetrievalEvalResult(
        cases=len(cases),
        recall_at_k={k: recall_totals[k] / total for k in k_values},
        hit_rate_at_k={k: hit_totals[k] / total for k in k_values},
        mrr=reciprocal_rank_total / total,
    )


def result_to_dict(result: RetrievalEvalResult) -> dict[str, Any]:
    """Convert eval result to a JSON-friendly dict."""

    return {
        "cases": result.cases,
        "recall_at_k": {str(k): value for k, value in result.recall_at_k.items()},
        "hit_rate_at_k": {str(k): value for k, value in result.hit_rate_at_k.items()},
        "mrr": result.mrr,
    }


def _match_positions(hits: list[SearchHit], relevant: list[str]) -> list[int]:
    positions: list[int] = []
    remaining = set(relevant)
    for position, hit in enumerate(hits, start=1):
        matched = [item for item in remaining if _hit_matches(hit, item)]
        if matched:
            positions.append(position)
            for item in matched:
                remaining.discard(item)
    return positions


def _hit_matches(hit: SearchHit, expected: str) -> bool:
    needle = expected.lower()
    chunk = hit.chunk
    fields = [
        chunk.chunk_id,
        chunk.source_file,
        chunk.heading or "",
        chunk.text,
    ]
    return any(needle in field.lower() for field in fields)
