"""LLM-as-judge for sparse-only vs sparse+dense retrieval comparisons.

Workflow:
1) Run comparison (reranker forced OFF inside that script):
   python eval/compare_sparse_vs_dense.py --file eval/queries_eval_10_mens.txt

2) Judge:
   python eval/judge_retrieval_outputs.py
   python eval/judge_retrieval_outputs.py --input eval/artifacts/raw/compare_results_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.openai_client import llm_judge_retrieval, openai_is_configured


def _latest_compare_result(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("compare_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to compare_results.json (defaults to latest in eval/results/)")
    parser.add_argument("--output", type=str, help="Path to write judge_results.json (defaults to eval/results/judge_results_<timestamp>.json)")
    args = parser.parse_args()

    if not openai_is_configured():
        raise SystemExit("OpenAI is not configured. Set OPENAI_API_KEY to run the judge.")

    raw_dir = PROJECT_ROOT / "eval" / "artifacts" / "raw"
    judged_dir = PROJECT_ROOT / "eval" / "artifacts" / "judged"
    input_path: Path
    if args.input:
        input_path = Path(args.input)
    else:
        latest = _latest_compare_result(raw_dir)
        if latest is None:
            raise SystemExit("No compare_results_*.json found in eval/artifacts/raw/. Run eval/compare_sparse_vs_dense.py first.")
        input_path = latest

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = judged_dir / f"judge_results_{stamp}.json"

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    judged = []

    for item in raw:
        query = item["query"]
        sparse_only = item["sparse_only"]
        sparse_plus_dense = item["sparse_plus_dense"]

        # Judge only the retrieval outcome. Explanations are present but should not be scored for prose.
        result = llm_judge_retrieval(
            user_query=query,
            sparse_only={"top_outfits": sparse_only.get("top_outfits", []), "parsed_constraints": sparse_only.get("parsed_constraints", {})},
            sparse_plus_dense={"top_outfits": sparse_plus_dense.get("top_outfits", []), "parsed_constraints": sparse_plus_dense.get("parsed_constraints", {})},
        )
        judged.append(
            {
                "query": query,
                "judge_result": result,
                "judge_status": "ok" if result is not None else "null_response",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(judged, indent=2), encoding="utf-8")
    print(f"Read {len(raw)} comparisons from {input_path}")
    print(f"Wrote {len(judged)} judgments to {output_path}")


if __name__ == "__main__":
    main()

