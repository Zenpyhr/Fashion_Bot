"""LLM-as-judge for a single retrieval evaluation run.

Workflow:
1) Run (reranker forced OFF inside that script):
   python eval/recommender_eval/run_retrieval_eval.py --file eval/recommender_eval/queries_eval_10_mens.txt

2) Judge the latest run:
   python eval/recommender_eval/judge_retrieval_run.py
   python eval/recommender_eval/judge_retrieval_run.py --input eval/recommender_eval/artifacts/raw/run_YYYYMMDD_HHMMSS.json
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

from src.integrations.openai_client import llm_score_retrieval, openai_is_configured


def _latest_run_result(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Path to run_*.json (defaults to latest in eval/recommender_eval/artifacts/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write judge_results.json (defaults to eval/recommender_eval/artifacts/judged/judge_results_<timestamp>.json)",
    )
    args = parser.parse_args()

    if not openai_is_configured():
        raise SystemExit("OpenAI is not configured. Set OPENAI_API_KEY to run the judge.")

    raw_dir = EVAL_DIR / "artifacts" / "raw"
    judged_dir = EVAL_DIR / "artifacts" / "judged"

    if args.input:
        input_path = Path(args.input)
    else:
        latest = _latest_run_result(raw_dir)
        if latest is None:
            raise SystemExit(
                "No run_*.json found in eval/recommender_eval/artifacts/raw/. "
                "Run eval/recommender_eval/run_retrieval_eval.py first."
            )
        input_path = latest

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = judged_dir / f"judge_results_{stamp}.json"

    raw_payload = json.loads(input_path.read_text(encoding="utf-8"))
    raw_items = raw_payload.get("items", raw_payload)

    judged_items = []
    sum_overall = 0
    sum_relevance = 0
    sum_constraint_fit = 0
    sum_coherence = 0
    scored_count = 0

    for item in raw_items:
        query = item.get("query", "")
        result_obj = item.get("result", {}) or {}

        judge_in = {
            "parsed_constraints": result_obj.get("parsed_constraints", {}),
            "candidate_pools": result_obj.get("candidate_pools", {}),
            "top_outfits": result_obj.get("top_outfits", []),
            "llm_status": result_obj.get("llm_status"),
        }

        judge_result = llm_score_retrieval(user_query=query, retrieval_output=judge_in)

        derived = {"overall": None, "relevance": None, "constraint_fit": None, "coherence": None}
        if isinstance(judge_result, dict):
            scores = judge_result.get("scores") or {}
            if isinstance(scores, dict):
                try:
                    derived["relevance"] = int(scores.get("relevance")) if scores.get("relevance") is not None else None
                    derived["constraint_fit"] = int(scores.get("constraint_fit")) if scores.get("constraint_fit") is not None else None
                    derived["coherence"] = int(scores.get("coherence")) if scores.get("coherence") is not None else None
                    derived["overall"] = int(scores.get("overall")) if scores.get("overall") is not None else None
                except Exception:
                    derived = {"overall": None, "relevance": None, "constraint_fit": None, "coherence": None}

        if derived["overall"] is not None:
            scored_count += 1
            sum_overall += derived["overall"]
            sum_relevance += derived["relevance"] or 0
            sum_constraint_fit += derived["constraint_fit"] or 0
            sum_coherence += derived["coherence"] or 0

        judged_items.append(
            {
                "query": query,
                "raw_output": result_obj,
                "judge_result": judge_result,
                "judge_status": "ok" if judge_result is not None else "null_response",
                "derived": derived,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_run_file": str(input_path),
        "config_snapshot": raw_payload.get("config_snapshot"),
        "count": len(raw_items),
        "scored_count": scored_count,
        "sum_overall": sum_overall,
        "mean_overall": (sum_overall / scored_count) if scored_count else None,
        "mean_subscores": {
            "relevance": (sum_relevance / scored_count) if scored_count else None,
            "constraint_fit": (sum_constraint_fit / scored_count) if scored_count else None,
            "coherence": (sum_coherence / scored_count) if scored_count else None,
        },
    }

    output_payload = {"summary": summary, "items": judged_items}
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Read {len(raw_items)} run items from {input_path}")
    print(f"Wrote {len(judged_items)} judgments to {output_path}")


if __name__ == "__main__":
    main()

