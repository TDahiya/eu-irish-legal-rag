# RRF weight sweep

`python -m eval.sweep_rrf`. Weights multiply each retriever's
reciprocal-rank contribution: `score(d) = sum of w / (k + rank)`.
Relevance is judged at instrument level over the 60 answerable goldset
questions.

| bm25 : dense | recall@5 | recall@10 | MRR | nDCG@10 | citation R@10 | conceptual R@10 |
|---|---|---|---|---|---|---|
| 1 : 1 | 90.0% | 93.3% | 0.843 | 0.838 | 100.0% | 87.1% |
| 1.5 : 1 | 90.0% | 91.7% | 0.852 | 0.848 | 100.0% | 87.1% |
| 2 : 1 | 90.0% | 91.7% | 0.849 | 0.846 | 100.0% | 87.1% |
| 3 : 1 | 90.0% | 91.7% | 0.861 | 0.853 | 100.0% | 87.1% |
| 4 : 1 | 90.0% | 90.0% | 0.868 | 0.850 | 100.0% | 83.9% |
| 1 : 1.5 | 86.7% | 88.3% | 0.821 | 0.814 | 93.8% | 83.9% |
| 1 : 2 | 86.7% | 88.3% | 0.817 | 0.810 | 93.8% | 83.9% |
| 2 : 0.5 | 90.0% | 90.0% | 0.868 | 0.850 | 100.0% | 83.9% |

**Equal weights win, which was not the expectation.** Fusion visibly cost
BM25 its best citation answers, so weighting lexical higher looked free.
It is not: 3:1 lifts MRR from 0.843 to 0.861 but drops recall@10 from
93.3% to 91.7%. All eight top chunks are put in front of the generator, so
a relevant chunk being in the window is worth more than its position
inside it, and recall is the metric to maximise.

Up-weighting dense is worse on every measure. That is the more interesting
finding: on legal text the lexical retriever carries more of the load than
the usual 'embeddings beat keywords' story suggests.
