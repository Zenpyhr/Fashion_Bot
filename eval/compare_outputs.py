"""Compare two saved benchmark result files at a high level."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_results(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(payload: dict) -> dict[str, dict]:
    return {entry["id"]: entry for entry in payload["results"]}


def summarize_top_outfit(entry: dict) -> str:
    outfits = entry["result"].get("outfits", [])
    if not outfits:
        return "NO OUTFIT"
    top_outfit = outfits[0]
    items = top_outfit.get("items", [])
    categories = [str(item.get("normalized_category")) for item in items]
    return " + ".join(categories)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python eval/compare_outputs.py <old_results.json> <new_results.json>")

    old_payload = load_results(sys.argv[1])
    new_payload = load_results(sys.argv[2])

    old_index = build_index(old_payload)
    new_index = build_index(new_payload)

    shared_ids = sorted(set(old_index) & set(new_index))
    print(f"Comparing {len(shared_ids)} shared benchmark queries")
    print()

    for query_id in shared_ids:
        old_entry = old_index[query_id]
        new_entry = new_index[query_id]
        print(f"{query_id}: {old_entry['query']}")
        print(f"  old top outfit: {summarize_top_outfit(old_entry)}")
        print(f"  new top outfit: {summarize_top_outfit(new_entry)}")
        print()


if __name__ == "__main__":
    main()
