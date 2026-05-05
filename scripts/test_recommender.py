"""Run one recommendation query and print a compact human-readable summary."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.openai_client import openai_is_configured
from src.recommender.outfits import build_outfits
from src.shared.config import settings


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/test_recommender.py "your query here"')

    query = " ".join(sys.argv[1:])
    result = build_outfits(query)

    print("LLM configured:", openai_is_configured())
    print("LLM status:", result.get("llm_status"))
    print("Dense rerank enabled:", settings.enable_dense_retrieval_rerank)
    print("Query:", query)
    print()
    print("Parsed constraints:")
    for key, value in result["parsed_constraints"].items():
        print(f"  {key}: {value}")

    print()
    print("Top outfits:")
    for index, outfit in enumerate(result["outfits"], start=1):
        categories = " + ".join(str(item["normalized_category"]) for item in outfit["items"])
        colors = ", ".join(str(item["normalized_color"]) for item in outfit["items"])
        print(f"  #{index}: {categories} | score={outfit['score']}")
        print(f"     colors: {colors}")
        print(f"     explanation: {outfit['explanation']}")


if __name__ == "__main__":
    main()
