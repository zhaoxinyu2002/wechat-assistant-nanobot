# Nanobot Agentic RAG Benchmarks

This document summarizes the local benchmark results for the knowledge-base retrieval pipeline used by the WeChat Agentic RAG assistant. The benchmark focuses on retrieval quality after documents have been ingested into the local knowledge base.

## Metrics

- `Recall@K`: whether the expected source document appears in the top-K retrieved results.
- `MRR`: mean reciprocal rank of the expected source document. Higher means the correct source is ranked earlier.
- `Avg query ms`: average retrieval latency per query in the local benchmark run.

## 1. CMRC2018 Chinese Retrieval

- Dataset: CMRC2018 dev subset
- Scale: 150 Chinese paragraph documents, 300 questions
- Task: retrieve the original paragraph document for each question
- Pipeline: text ingestion -> chunking -> embedding/search -> Recall@K and MRR

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg query ms |
|---|---:|---:|---:|---:|---:|---:|
| Hashing Keyword | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.21 |
| Hashing Hybrid | 0.4100 | 0.6067 | 0.6933 | 0.7900 | 0.5300 | 74.72 |
| BGE-M3 Vector | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 | 44.04 |
| BGE-M3 Hybrid | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 | 109.01 |
| BGE-M3 + Reranker | 0.9767 | 0.9933 | 0.9933 | 0.9933 | 0.9850 | 440.69 |

Compared with the hashing hybrid baseline, BGE-M3 plus reranking improves Recall@1 from 41.00% to 97.67% and MRR from 0.5300 to 0.9850.

## 2. UDA-Benchmark Paper Evidence Retrieval

- Dataset: UDA-Benchmark `paper_text` evidence subset
- Scale: 150 paper evidence documents, 300 paper questions
- Task: retrieve the original paper evidence document for each question
- Scope: evaluates retrieval over provided evidence text, not PDF parsing

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg query ms |
|---|---:|---:|---:|---:|---:|---:|
| Hashing Hybrid | 0.0400 | 0.0833 | 0.1067 | 0.1733 | 0.0745 | 78.15 |
| BGE-M3 Vector | 0.4033 | 0.4733 | 0.5400 | 0.6100 | 0.4583 | 48.75 |
| BGE-M3 Hybrid | 0.1000 | 0.2033 | 0.2767 | 0.4133 | 0.1795 | 118.99 |
| BGE-M3 Hybrid + Reranker | 0.4133 | 0.4900 | 0.5367 | 0.5767 | 0.4601 | 451.53 |

Compared with the hashing hybrid baseline, the best reproduced UDA evidence result improves Recall@1 from 4.00% to 41.33% and MRR from 0.0745 to 0.4601.

The UDA result also shows that the current fixed hybrid weighting can hurt paper-domain retrieval, where keyword matching introduces noise. BGE-M3 vector search is the better default for this subset.

## 3. Benchmark Scope

- The benchmark results above evaluate retrieval quality after document text has been ingested.
- The project also supports PDF ingestion through MinerU-based structured parsing and OCR.
- End-to-end PDF parsing quality is treated as an ingestion capability rather than the primary retrieval benchmark target.
- The current quantified results focus on reproducible text/evidence retrieval tasks:
  - CMRC2018 Chinese text retrieval
  - UDA-Benchmark paper evidence retrieval

## Summary

- BGE-M3 and BGE-reranker are the main quantified retrieval improvements in this project.
- On CMRC2018, BGE-M3 plus reranking improves Recall@1 from 41.00% to 97.67% and MRR from 0.5300 to 0.9850 over the hashing hybrid baseline.
- On UDA paper evidence retrieval, the best reproduced setting improves Recall@1 from 4.00% to 41.33% and MRR from 0.0745 to 0.4601 over the hashing hybrid baseline.
- The UDA experiment suggests retrieval mode should be selected by domain: vector retrieval performs better than fixed-weight hybrid retrieval on this paper-evidence subset.
