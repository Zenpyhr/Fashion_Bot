# QA Evaluation (Track A)

This folder evaluates the QA/RAG pipeline with fixed fashion questions in `evaluation_qa.py`.

## What the evaluation does

For each test case:
1. Retrieve context chunks with `query_answer.retrieve(...)`.
2. Generate an answer from the retrieved context.
3. Score retrieval quality against `expected_scopes`.
4. Use an LLM judge to score whether the answer is grounded in the retrieved context.

## Metrics used

- **Recall@k (`recall_at_k`)**  
  Fraction of expected scopes that appear in the retrieved set.

- **Precision@k (`precision_at_k`)**  
  Fraction of retrieved scopes that are correct (in expected scopes).

- **MRR (`mrr`)**  
  Reciprocal rank of the first retrieved chunk whose scope matches expected scopes.

- **Judge score (`judge_score`) / Faithfulness**  
  LLM grounding score on a 0–2 scale:
  - `0`: mostly unsupported / major hallucination
  - `1`: partially supported
  - `2`: strongly supported by retrieved evidence

The script also reports:
- `mode` (retrieval mode from `query_answer.retrieve`)
- `retrieved_count` (number of returned context chunks)

## Aggregate summary

At the end, the script prints averages:
- `average_recall_at_k`
- `average_precision_at_k`
- `mrr` (average rr)
- `faithfulness` (average judge score)