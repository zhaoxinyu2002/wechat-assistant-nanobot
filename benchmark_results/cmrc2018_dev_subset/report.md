# CMRC2018 dev subset retrieval benchmark

- Dataset: CMRC2018 dev subset
- Documents: 150 paragraphs as .txt files
- Queries: 300 questions
- Relevant label: original paragraph file should appear in top-k retrieval results

| Experiment | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg query ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_hashing_keyword | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.06 |
| baseline_hashing_hybrid | 0.4100 | 0.6067 | 0.6933 | 0.7900 | 0.5300 | 82.20 |
| ours_bge_vector | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 | 46.12 |
| ours_bge_hybrid | 0.9700 | 0.9900 | 0.9900 | 0.9933 | 0.9805 | 141.39 |
| ours_bge_hybrid_rerank | 0.9767 | 0.9933 | 0.9933 | 0.9933 | 0.9850 | 456.57 |
