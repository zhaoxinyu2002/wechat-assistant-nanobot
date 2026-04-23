# Nanobot RAG Benchmarks

This document summarizes the local benchmark results for the enhanced knowledge-base retrieval pipeline.

## Metrics

- `Recall@K`: whether the expected source document appears in the top-K retrieved results.
- `MRR`: mean reciprocal rank of the expected source document. Higher means the correct source is ranked earlier.

## 1. CMRC2018 Chinese Retrieval

- Dataset: CMRC2018 dev subset
- Scale: 150 Chinese paragraph documents, 300 questions
- Task: retrieve the original paragraph document for each question
- Pipeline: text ingestion -> chunking -> embedding/search -> Recall@K and MRR

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Hashing Keyword | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hashing Hybrid | 0.4100 | 0.6067 | 0.6933 | 0.7900 | 0.5300 |
| BGE-M3 Vector | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 |
| BGE-M3 Hybrid | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 |
| BGE-M3 + Reranker | 0.9767 | 0.9933 | 0.9933 | 0.9933 | 0.9850 |

Compared with the hashing hybrid baseline, BGE-M3 plus reranking improves Recall@1 from 41.00% to 97.67% and MRR from 0.5300 to 0.9850.

## 2. UDA-Benchmark Paper Evidence Retrieval

- Dataset: UDA-Benchmark `paper_text` evidence subset
- Scale: 150 paper evidence documents, 300 paper questions
- Task: retrieve the original paper evidence document for each question
- Scope: evaluates retrieval over provided evidence text, not PDF parsing

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Hashing Hybrid | 0.0400 | 0.0833 | 0.1067 | 0.1733 | 0.0745 |
| BGE-M3 Vector | 0.4033 | 0.4733 | 0.5400 | 0.6100 | 0.4583 |
| BGE-M3 Hybrid | 0.1000 | 0.2033 | 0.2767 | 0.4133 | 0.1795 |
| BGE-M3 Hybrid + Reranker | 0.4133 | 0.4900 | 0.5367 | 0.5767 | 0.4601 |

Compared with the hashing hybrid baseline, the best reproduced UDA evidence result improves Recall@1 from 4.00% to 41.33% and MRR from 0.0745 to 0.4601.

The UDA result also shows that the current fixed hybrid weighting can hurt paper-domain retrieval, where keyword matching introduces noise. BGE-M3 vector search is the better default for this subset.

## 3. PDF Ingestion Note

- The project includes a PDF ingestion path and MinerU integration for structured PDF parsing.
- In practice, large-batch PDF parsing was much more expensive and less stable than text/evidence retrieval benchmarking on the current machine.
- For this reason, the benchmark focus is intentionally placed on the two reproducible retrieval evaluations above:
  - CMRC2018 Chinese text retrieval
  - UDA-Benchmark paper evidence retrieval
- MinerU remains part of the system as an engineering capability for PDF parsing and knowledge-base ingestion, but it is not the primary quantified result in the current benchmark summary.

## Summary

- BGE-M3 and BGE-reranker are the main quantified improvements in this project.
- The strongest reproducible results come from Chinese text retrieval and paper-domain evidence retrieval.
- PDF parsing support exists in the codebase, but the current project summary focuses on the stable, repeatable RAG retrieval benchmarks above.
