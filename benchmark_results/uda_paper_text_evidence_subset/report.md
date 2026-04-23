# UDA-Benchmark paper text evidence retrieval benchmark

- Dataset: UDA-Benchmark paper_text QA subset
- Documents: 150 papers represented by gold evidence snippets
- Queries: 300 paper questions
- Relevant label: original paper file should appear in top-k retrieval results
- Note: this measures retrieval on paper evidence text, not end-to-end PDF parsing.

| Experiment | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg query ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_hashing_hybrid | 0.0400 | 0.0833 | 0.1067 | 0.1733 | 0.0745 | 78.15 |
| ours_bge_vector | 0.4033 | 0.4733 | 0.5400 | 0.6100 | 0.4583 | 48.75 |
| ours_bge_hybrid | 0.1000 | 0.2033 | 0.2767 | 0.4133 | 0.1795 | 118.99 |
| ours_bge_hybrid_rerank | 0.4133 | 0.4900 | 0.5367 | 0.5767 | 0.4601 | 451.53 |
