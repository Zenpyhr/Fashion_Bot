"""Build a smaller demo image subset from the full H&M image archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CATALOG_CSV = PROJECT_ROOT / "data" / "processed" / "catalog_items" / "catalog_items_mvp.csv"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "hm" / "images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "demo_images"
DEFAULT_FILTERED_CSV = PROJECT_ROOT / "data" / "processed" / "catalog_items" / "catalog_items_demo.csv"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "data" / "processed" / "catalog_items" / "catalog_items_demo_summary.json"
DEFAULT_EXCLUDED_CATEGORIES = {"flip_flop", "flats", "leggings_tights"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy only the images needed for a smaller demo subset.")
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG_CSV)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FILTERED_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--target-groups", nargs="*", default=["women"], help="Example: --target-groups women men")
    parser.add_argument("--roles", nargs="*", default=None, help="Example: --roles top bottom shoes outerwear")
    parser.add_argument("--categories", nargs="*", default=None, help="Example: --categories blouse shirt trousers skirt boots sneakers")
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDED_CATEGORIES),
        help="Categories to exclude from the demo subset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to keep after filtering.")
    parser.add_argument(
        "--sampling-strategy",
        choices=["balanced", "proportional"],
        default="balanced",
        help="How to reduce the filtered catalog when --limit is set. Default: balanced",
    )
    parser.add_argument(
        "--proportional-by",
        default="normalized_category",
        help="Column used for proportional sampling when --limit is set. Default: normalized_category",
    )
    parser.add_argument(
        "--skip-image-copy",
        action="store_true",
        help="Only build the filtered CSV and summary. Do not copy image files yet.",
    )
    return parser.parse_args()


def expected_image_relative_path(item_id: str) -> Path:
    prefix = item_id[:3]
    return Path(prefix) / f"{item_id}.jpg"


def filter_catalog(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()

    if args.target_groups:
        filtered = filtered[filtered["target_group"].isin(args.target_groups)].copy()

    if args.roles:
        filtered = filtered[filtered["recommendation_role"].isin(args.roles)].copy()

    if args.categories:
        filtered = filtered[filtered["normalized_category"].isin(args.categories)].copy()

    if args.exclude_categories:
        filtered = filtered[~filtered["normalized_category"].isin(args.exclude_categories)].copy()

    base_filtered = filtered.copy()
    filtered["item_id"] = filtered["item_id"].astype(str).str.zfill(10)
    filtered["image_relative_path"] = filtered["item_id"].map(lambda item_id: str(expected_image_relative_path(item_id)))
    filtered["image_path"] = filtered["image_relative_path"].map(lambda rel: str(args.output_root / rel))
    if args.limit is not None and len(filtered) > args.limit:
        if args.sampling_strategy == "balanced":
            filtered = balanced_sample(filtered, args.limit)
        else:
            filtered = proportional_sample(filtered, args.limit, args.proportional_by)

    # Enforce the exclusion list again after sampling, then top up from the clean remainder if needed.
    if args.exclude_categories:
        filtered = filtered[~filtered["normalized_category"].isin(args.exclude_categories)].copy()

    if args.limit is not None and len(filtered) < args.limit:
        selected_ids = set(filtered["item_id"].astype(str))
        remainder_df = base_filtered[~base_filtered["item_id"].astype(str).isin(selected_ids)].copy()
        extra_needed = args.limit - len(filtered)
        filtered = pd.concat([filtered, remainder_df.head(extra_needed)], ignore_index=True)

    return filtered


def proportional_sample(df: pd.DataFrame, limit: int, group_column: str) -> pd.DataFrame:
    """Keep rows proportionally across a subgroup column using a deterministic largest-remainder allocation."""

    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in filtered catalog.")

    if limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    group_sizes = df[group_column].value_counts(dropna=False).sort_index()
    total_rows = int(group_sizes.sum())
    if total_rows <= limit:
        return df.copy()

    exact_allocations = (group_sizes / total_rows) * limit
    base_allocations = exact_allocations.astype(int)
    remainder = exact_allocations - base_allocations

    allocated = int(base_allocations.sum())
    remaining_slots = limit - allocated

    # Distribute the leftover slots to the groups with the largest fractional remainders.
    if remaining_slots > 0:
        for group_value in remainder.sort_values(ascending=False).index[:remaining_slots]:
            base_allocations[group_value] += 1

    sampled_parts = []
    for group_value, keep_count in base_allocations.items():
        if keep_count <= 0:
            continue
        group_df = df[df[group_column] == group_value].copy()
        sampled_parts.append(group_df.head(int(keep_count)))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    # Guard against underfill from very small groups by topping up from the remaining rows.
    if len(sampled_df) < limit:
        selected_ids = set(sampled_df["item_id"].astype(str))
        remainder_df = df[~df["item_id"].astype(str).isin(selected_ids)].copy()
        extra_needed = limit - len(sampled_df)
        sampled_df = pd.concat([sampled_df, remainder_df.head(extra_needed)], ignore_index=True)

    return sampled_df


def _allocate_quota(counts: pd.Series, limit: int) -> dict[str, int]:
    """Allocate a balanced quota with caps, then redistribute leftover slots as evenly as possible."""

    groups = list(counts.index)
    group_count = len(groups)
    if group_count == 0:
        return {}

    allocation = {group: 0 for group in groups}
    remaining_slots = limit
    active_groups = set(groups)

    while remaining_slots > 0 and active_groups:
        share = max(1, remaining_slots // len(active_groups))
        progressed = False

        for group in list(active_groups):
            capacity = int(counts[group]) - allocation[group]
            if capacity <= 0:
                active_groups.remove(group)
                continue

            add_now = min(share, capacity, remaining_slots)
            if add_now <= 0:
                continue

            allocation[group] += add_now
            remaining_slots -= add_now
            progressed = True

            if allocation[group] >= int(counts[group]):
                active_groups.remove(group)
            if remaining_slots == 0:
                break

        if not progressed:
            break

    return allocation


def balanced_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Balance the demo subset across outfit roles first, then across categories within each role."""

    role_counts = df["recommendation_role"].value_counts()
    role_allocation = _allocate_quota(role_counts, limit)

    sampled_parts = []
    for role, role_limit in role_allocation.items():
        role_df = df[df["recommendation_role"] == role].copy()
        role_sample = proportional_sample(role_df, role_limit, "normalized_category")
        sampled_parts.append(role_sample)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled_df) < limit:
        selected_ids = set(sampled_df["item_id"].astype(str))
        remainder_df = df[~df["item_id"].astype(str).isin(selected_ids)].copy()
        sampled_df = pd.concat([sampled_df, remainder_df.head(limit - len(sampled_df))], ignore_index=True)

    return sampled_df


def copy_images(filtered_df: pd.DataFrame, source_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    for item_id in filtered_df["item_id"].drop_duplicates():
        relative_path = expected_image_relative_path(item_id)
        source_path = source_root / relative_path
        target_path = output_root / relative_path

        if not source_path.exists():
            missing.append(str(relative_path))
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            copied += 1

    return {
        "copied_count": copied,
        "missing_count": len(missing),
        "missing_examples": missing[:25],
    }


def build_summary(filtered_df: pd.DataFrame, copy_result: dict, args: argparse.Namespace) -> dict:
    return {
        "source_catalog_csv": str(args.catalog_csv),
        "source_images_root": str(args.source_root),
        "output_images_root": str(args.output_root),
        "row_count": int(len(filtered_df)),
        "unique_item_count": int(filtered_df["item_id"].nunique()),
        "target_group_counts": {
            str(key): int(value) for key, value in filtered_df["target_group"].value_counts().to_dict().items()
        },
        "role_counts": {
            str(key): int(value) for key, value in filtered_df["recommendation_role"].value_counts().to_dict().items()
        },
        "category_counts": {
            str(key): int(value)
            for key, value in filtered_df["normalized_category"].value_counts().head(20).to_dict().items()
        },
        **copy_result,
    }


def main() -> None:
    args = parse_args()

    if not args.catalog_csv.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {args.catalog_csv}")
    if not args.skip_image_copy and not args.source_root.exists():
        raise FileNotFoundError(f"Source image root not found: {args.source_root}")

    df = pd.read_csv(args.catalog_csv, dtype={"item_id": "string"})
    filtered_df = filter_catalog(df, args)
    if args.skip_image_copy:
        copy_result = {
            "copied_count": 0,
            "missing_count": 0,
            "missing_examples": [],
            "image_copy_skipped": True,
        }
    else:
        copy_result = copy_images(filtered_df, args.source_root, args.output_root)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output_csv, index=False)

    summary = build_summary(filtered_df, copy_result, args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Unique items: {filtered_df['item_id'].nunique()}")
    print(f"Copied images: {copy_result['copied_count']}")
    print(f"Missing images: {copy_result['missing_count']}")
    print(f"Filtered CSV: {args.output_csv}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Demo image root: {args.output_root}")


if __name__ == "__main__":
    main()
