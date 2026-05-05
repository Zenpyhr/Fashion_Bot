"""Compare sparse-only vs sparse+dense retrieval for a list of queries.

This script does NOT run automatically. It's a manual eval helper.

Usage:
  python eval/compare_sparse_vs_dense.py "query here"
  python eval/compare_sparse_vs_dense.py --file eval/queries_dense_eval.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.outfits import build_outfits
from src.shared.config import settings


def _load_queries_from_file(path: Path) -> list[str]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


def _summarize(result: dict) -> dict:
    outfits = result.get("outfits", [])
    return {
        "llm_status": result.get("llm_status"),
        "parsed_constraints": result.get("parsed_constraints"),
        "top_outfits": [
            {
                "score": outfit.get("score"),
                "signature": " + ".join(str(item.get("normalized_category")) for item in outfit.get("items", [])),
                "colors": [item.get("normalized_color") for item in outfit.get("items", [])],
                "explanation": outfit.get("explanation"),
            }
            for outfit in outfits
        ],
    }


def compare_query(query: str) -> dict:
    # Snapshot current setting, then compare outputs.
    old = settings.enable_dense_retrieval_rerank

    settings.enable_dense_retrieval_rerank = False
    sparse = _summarize(build_outfits(query))

    settings.enable_dense_retrieval_rerank = True
    dense = _summarize(build_outfits(query))

    settings.enable_dense_retrieval_rerank = old

    return {"query": query, "sparse_only": sparse, "sparse_plus_dense": dense}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--file", type=str, help="Path to newline-delimited query file")
    args = parser.parse_args()

    queries: list[str] = []
    if args.file:
        queries.extend(_load_queries_from_file(Path(args.file)))
    if args.query:
        queries.append(" ".join(args.query))

    if not queries:
        raise SystemExit("Provide a query or --file with queries.")

    results = [compare_query(q) for q in queries]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

