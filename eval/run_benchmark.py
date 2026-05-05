"""Run the recommender on a fixed benchmark set and save outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.outfits import build_outfits

BENCHMARK_PATH = PROJECT_ROOT / "eval" / "benchmark_queries.json"
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_benchmark() -> dict:
    queries = load_queries(BENCHMARK_PATH)
    results = []

    for query_entry in queries:
        result = build_outfits(query_entry["query"])
        results.append(
            {
                "id": query_entry["id"],
                "query": query_entry["query"],
                "tags": query_entry.get("tags", []),
                "result": result,
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(results),
        "results": results,
    }


def write_output(payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = OUTPUT_DIR / f"benchmark_results_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    payload = run_benchmark()
    output_path = write_output(payload)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
