# Nanobot RAG and PDF Parsing Benchmarks

This document summarizes the local benchmark results for the enhanced knowledge-base pipeline.

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
| BGE-M3 Vector + Reranker | 0.4133 | 0.5033 | 0.5467 | 0.6233 | 0.4726 |

Compared with the hashing hybrid baseline, BGE-M3 vector search plus reranking improves Recall@1 from 4.00% to 41.33% and MRR from 0.0745 to 0.4726.

The UDA result also shows that the current fixed hybrid weighting can hurt paper-domain retrieval, where keyword matching introduces noise. BGE-M3 vector search is the better default for this subset.

## 3. UDA-Benchmark PDF Parsing with MinerU

- Dataset: UDA-Benchmark `paper_text` PDF subset
- Scale: 5 small real paper PDFs, 21 paper questions
- Task: start from PDF files and retrieve the original PDF for each question
- Pipeline: PDF -> parser -> chunking -> BGE-M3 embedding -> vector search
- Scope: first 6 pages per PDF for a safe small-sample test

Chunk-level retrieval:

| Method | Parsed Files | Text Chars | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic PDF Fallback | 5/5 | 257,163 | 17 | 0.2857 | 0.6667 | 0.8571 | 0.4817 |
| MinerU | 5/5 | 62,554 | 134 | 0.6667 | 0.6667 | 0.7619 | 0.6905 |

Document-level retrieval after collapsing multiple chunks from the same PDF:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Basic PDF Fallback | 0.2857 | 0.7143 | 1.0000 | 0.5190 |
| MinerU | 0.6667 | 0.7619 | 0.9524 | 0.7492 |

MinerU improves PDF-RAG retrieval by producing cleaner, more structured chunks. In this small-sample PDF benchmark, chunk-level Recall@1 improves from 28.57% to 66.67%, and MRR improves from 0.4817 to 0.6905.

## Summary

- BGE-M3 and BGE-reranker mainly improve semantic retrieval quality.
- MinerU mainly improves PDF parsing quality and chunk structure for PDF-RAG.
- The benchmark suite covers Chinese text retrieval, paper-domain evidence retrieval, and real PDF end-to-end retrieval.
