"""Run a retrieval evaluation (current config) for a list of queries.

This script produces ONE raw artifact for the current code/config, so you can
track the judge score over time as you make changes.

Usage:
  python eval/recommender_eval/run_retrieval_eval.py --file eval/recommender_eval/queries_eval_10_mens.txt
  python eval/recommender_eval/run_retrieval_eval.py --enable-reranker true --file eval/recommender_eval/queries_eval_10_mens.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
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
        "candidate_pools": result.get("candidate_pools", {}),
        "top_outfits": [
            {
                "score": outfit.get("score"),
                "signature": " + ".join(str(item.get("normalized_category")) for item in outfit.get("items", [])),
                "colors": [item.get("normalized_color") for item in outfit.get("items", [])],
                "explanation": outfit.get("explanation"),
                "items": [
                    {
                        "item_id": item.get("item_id"),
                        "display_name": item.get("display_name"),
                        "role": item.get("recommendation_role"),
                        "normalized_category": item.get("normalized_category"),
                        "normalized_color": item.get("normalized_color"),
                        "section_theme": item.get("section_theme"),
                        "score": item.get("score"),
                    }
                    for item in outfit.get("items", [])
                ],
            }
            for outfit in outfits
        ],
    }


def run_query(query: str) -> dict:
    result = _summarize(build_outfits(query))
    return {"query": query, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-reranker",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to enable the final OpenAI reranker during the run (default: false)",
    )
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--file", type=str, help="Path to newline-delimited query file")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write JSON results (defaults to eval/recommender_eval/artifacts/raw/run_<timestamp>.json)",
    )
    args = parser.parse_args()

    queries: list[str] = []
    if args.file:
        queries.extend(_load_queries_from_file(Path(args.file)))
    if args.query:
        queries.append(" ".join(args.query))
    if not queries:
        raise SystemExit("Provide a query or --file with queries.")

    old_reranker = settings.enable_openai_reranker
    settings.enable_openai_reranker = (args.enable_reranker == "true")
    try:
        results = [run_query(q) for q in queries]
    finally:
        settings.enable_openai_reranker = old_reranker

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = EVAL_DIR / "artifacts" / "raw" / f"run_{stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_snapshot": {
            "enable_dense_retrieval_rerank": settings.enable_dense_retrieval_rerank,
            "dense_shortlist_k_per_role": settings.dense_shortlist_k_per_role,
            "dense_rerank_n_per_role": settings.dense_rerank_n_per_role,
            "enable_openai_query_parser": settings.enable_openai_query_parser,
            "enable_openai_reranker": (args.enable_reranker == "true"),
            "openai_model_query_parser": settings.openai_model_query_parser,
            "openai_model_judge": settings.openai_model_judge,
            "openai_embedding_model": settings.openai_embedding_model,
        },
        "items": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

