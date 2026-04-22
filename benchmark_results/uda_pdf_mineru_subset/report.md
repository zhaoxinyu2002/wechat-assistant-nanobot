# UDA-Benchmark PDF MinerU parsing benchmark

- Dataset: UDA-Benchmark paper_text QA with real PDFs from paper_docs.zip
- PDFs: 5 small papers
- Queries: 21 paper questions
- Retrieval: BGE-M3 vector search
- Label: original PDF should appear in top-k retrieval results
- Scope: first 6 pages per PDF for a safe small-sample test

| Experiment | Parsed files | Text chars | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Ingest s | Avg query ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| without_mineru_basic_fallback | 5/5 | 257163 | 17 | 0.2857 | 0.6667 | 0.8571 | 0.4817 | 29.81 | 40.89 |
| with_mineru_agent | 5/5 | 62554 | 134 | 0.6667 | 0.6667 | 0.7619 | 0.6905 | 139.18 | 55.16 |

## Per-file parse stats

| Experiment | File | Parser | Sections | Chunks | Text chars | Readable ratio |
|---|---|---|---:|---:|---:|---:|
| without_mineru_basic_fallback | 1805.10824.pdf | basic-fallback | 1 | 2 | 61041 | 0.8503 |
| without_mineru_basic_fallback | 1909.11467.pdf | basic-fallback | 2 | 3 | 50600 | 0.8634 |
| without_mineru_basic_fallback | 1901.10133.pdf | basic-fallback | 2 | 4 | 50774 | 0.8490 |
| without_mineru_basic_fallback | 1608.01884.pdf | basic-fallback | 3 | 4 | 55665 | 0.8382 |
| without_mineru_basic_fallback | 1901.04899.pdf | basic-fallback | 2 | 4 | 39083 | 0.8454 |
| with_mineru_agent | 1805.10824.pdf | mineru-precision | 76 | 76 | 17874 | 0.9989 |
| with_mineru_agent | 1909.11467.pdf | mineru-precision | 9 | 9 | 11407 | 0.9957 |
| with_mineru_agent | 1901.10133.pdf | mineru-precision | 34 | 34 | 8637 | 0.9929 |
| with_mineru_agent | 1608.01884.pdf | mineru-precision | 9 | 9 | 15726 | 0.9853 |
| with_mineru_agent | 1901.04899.pdf | mineru-precision | 6 | 6 | 8910 | 0.9965 |

## Doc-level retrieval metrics

This collapses multiple chunks from the same PDF into one source document before computing Recall@K.

| Experiment | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| without_mineru_basic_fallback | 0.2857 | 0.7143 | 1.0000 | 0.5190 |
| with_mineru_agent | 0.6667 | 0.7619 | 0.9524 | 0.7492 |
