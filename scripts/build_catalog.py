"""Run the catalog ingestion pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.normalize_catalog import DEFAULT_OUTPUT_PATH, DEFAULT_SUMMARY_PATH, normalize_catalog


def main() -> None:
    df = normalize_catalog()
    print(f"Built normalized catalog: {len(df)} rows")
    print(f"CSV output: {DEFAULT_OUTPUT_PATH}")
    print(f"Summary output: {DEFAULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
