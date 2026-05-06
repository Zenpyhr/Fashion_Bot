"""Normalize raw H&M metadata into recommendation-friendly fields."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.recommender.ingest_catalog import DEFAULT_SOURCE_PATH, ingest_catalog
from src.shared.utils import slugify_name

ROLE_BY_PRODUCT_TYPE = {
    "blouse": "top",
    "bodysuit": "top",
    "hoodie": "top",
    "polo shirt": "top",
    "shirt": "top",
    "sweater": "top",
    "t-shirt": "top",
    "top": "top",
    "vest top": "top",
    "blazer": "outerwear",
    "cardigan": "outerwear",
    "coat": "outerwear",
    "jacket": "outerwear",
    "outdoor waistcoat": "outerwear",
    "tailored waistcoat": "outerwear",
    "leggings/tights": "bottom",
    "outdoor trousers": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "trousers": "bottom",
    "ballerinas": "shoes",
    "boots": "shoes",
    "flat shoe": "shoes",
    "flat shoes": "shoes",
    "flip flop": "shoes",
    "heeled sandals": "shoes",
    "heels": "shoes",
    "moccasins": "shoes",
    "other shoe": "shoes",
    "pumps": "shoes",
    "sandals": "shoes",
    "slippers": "shoes",
    "sneakers": "shoes",
    "wedge": "shoes",
}

NORMALIZED_CATEGORY_BY_PRODUCT_TYPE = {
    "t-shirt": "tshirt",
    "vest top": "tank_top",
    "polo shirt": "polo_shirt",
    "outdoor waistcoat": "waistcoat",
    "tailored waistcoat": "waistcoat",
    "flat shoe": "flats",
    "flat shoes": "flats",
}

PATTERN_MAP = {
    "all over pattern": "patterned",
    "application/3d": "applique",
    "check": "check",
    "colour blocking": "color_block",
    "denim": "denim",
    "dot": "dot",
    "embroidery": "embroidered",
    "front print": "print",
    "glittering/metallic": "metallic",
    "jacquard": "jacquard",
    "lace": "lace",
    "melange": "melange",
    "mixed solid/pattern": "mixed",
    "other pattern": "patterned",
    "other structure": "textured",
    "placement print": "print",
    "solid": "solid",
    "stripe": "stripe",
    "treatment": "treated",
}

DEFAULT_OUTPUT_PATH = Path("data/recommender/processed/catalog_items/catalog_items_mvp.csv")
DEFAULT_SUMMARY_PATH = Path("data/recommender/processed/catalog_items/catalog_items_mvp_summary.json")
ALLOWED_TARGET_GROUPS = {"women", "men"}
EXCLUDED_NORMALIZED_CATEGORIES = {"bodysuit", "other_shoe", "slippers"}

OUTPUT_COLUMNS = [
    "item_id",
    "source_type",
    "image_path",
    "display_name",
    "description",
    "target_group",
    "recommendation_role",
    "normalized_category",
    "product_family",
    "normalized_color",
    "color_detail",
    "color_tone",
    "normalized_pattern",
    "section_theme",
    "article_id",
    "product_code",
    "product_type_name",
    "product_group_name",
    "index_name",
    "index_group_name",
    "section_name",
    "department_name",
    "garment_group_name",
]


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_label(value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"unknown", "undefined"}:
        return None
    return lowered


def _map_recommendation_role(product_type_name: object) -> str | None:
    key = _clean_label(product_type_name)
    if key is None:
        return None
    return ROLE_BY_PRODUCT_TYPE.get(key)


def _map_normalized_category(product_type_name: object) -> str | None:
    key = _clean_label(product_type_name)
    if key is None:
        return None
    if key in NORMALIZED_CATEGORY_BY_PRODUCT_TYPE:
        return NORMALIZED_CATEGORY_BY_PRODUCT_TYPE[key]
    return slugify_name(key).replace("-", "_")


def _map_product_family(recommendation_role: object) -> str | None:
    role = _clean_label(recommendation_role)
    if role is None:
        return None
    family_map = {
        "top": "upper_body",
        "bottom": "lower_body",
        "outerwear": "outerwear",
        "shoes": "footwear",
    }
    return family_map.get(role)


def _map_pattern(value: object) -> str | None:
    key = _clean_label(value)
    if key is None:
        return None
    return PATTERN_MAP.get(key, slugify_name(key).replace("-", "_"))


def _map_target_group(index_group_name: object, index_name: object, section_name: object) -> str:
    index_group = _clean_label(index_group_name) or ""
    index_value = _clean_label(index_name) or ""
    section_value = _clean_label(section_name) or ""
    combined = " ".join(part for part in (index_value, section_value) if part)

    if index_group in {"ladieswear", "divided"}:
        return "women"
    if index_group == "menswear":
        return "men"
    if index_group == "baby/children":
        if "baby" in combined:
            return "baby"
        return "kids"
    if index_group == "sport":
        if "men" in combined:
            return "men"
        if "ladies" in combined or "women" in combined:
            return "women"
        if "kids" in combined or "girl" in combined or "boy" in combined:
            return "kids"
        return "unisex"
    return "unisex"


def _map_section_theme(section_name: object, department_name: object, index_group_name: object) -> str | None:
    section_value = _clean_label(section_name) or ""
    department_value = _clean_label(department_name) or ""
    index_group_value = _clean_label(index_group_name) or ""
    combined = " ".join(part for part in (section_value, department_value, index_group_value) if part)

    if "sport" in combined:
        return "sport"
    if "tailoring" in combined:
        return "tailoring"
    if "casual" in combined:
        return "casual"
    if "basic" in combined:
        return "basics"
    if "trend" in combined:
        return "trend"
    if "everyday" in combined:
        return "everyday"
    if not section_value:
        return None
    return slugify_name(section_value)


def normalize_catalog_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the H&M catalog to the MVP roles and engineer recommendation columns."""

    normalized = df.copy()
    normalized["product_type_name"] = normalized["product_type_name"].map(_clean_text)
    normalized["recommendation_role"] = normalized["product_type_name"].map(_map_recommendation_role)

    # Keep only the four outfit roles used in the MVP recommender.
    normalized = normalized[normalized["recommendation_role"].notna()].copy()

    normalized["item_id"] = normalized["article_id"].astype("string")
    normalized["source_type"] = "catalog"
    normalized["image_path"] = pd.NA
    normalized["display_name"] = normalized["prod_name"].map(_clean_text)
    normalized["description"] = normalized["detail_desc"].map(_clean_text)
    normalized["target_group"] = normalized.apply(
        lambda row: _map_target_group(row["index_group_name"], row["index_name"], row["section_name"]),
        axis=1,
    )
    normalized["normalized_category"] = normalized["product_type_name"].map(_map_normalized_category)
    normalized["product_family"] = normalized["recommendation_role"].map(_map_product_family)
    normalized["normalized_color"] = normalized["perceived_colour_master_name"].map(_clean_label)
    normalized["color_detail"] = normalized["colour_group_name"].map(_clean_label)
    normalized["color_tone"] = normalized["perceived_colour_value_name"].map(_clean_label)
    normalized["normalized_pattern"] = normalized["graphical_appearance_name"].map(_map_pattern)
    normalized["section_theme"] = normalized.apply(
        lambda row: _map_section_theme(row["section_name"], row["department_name"], row["index_group_name"]),
        axis=1,
    )

    # Keep the MVP focused on adult recommendations only.
    normalized = normalized[normalized["target_group"].isin(ALLOWED_TARGET_GROUPS)].copy()

    # Drop noisy categories that tend to produce awkward or ambiguous outfit results.
    normalized = normalized[~normalized["normalized_category"].isin(EXCLUDED_NORMALIZED_CATEGORIES)].copy()

    return normalized.loc[:, OUTPUT_COLUMNS].sort_values(
        by=["target_group", "recommendation_role", "normalized_category", "item_id"]
    )


def build_summary(df: pd.DataFrame) -> dict:
    """Build a small summary that helps validate the preprocessing output."""

    return {
        "row_count": int(len(df)),
        "role_counts": {str(key): int(value) for key, value in df["recommendation_role"].value_counts().to_dict().items()},
        "target_group_counts": {str(key): int(value) for key, value in df["target_group"].value_counts().to_dict().items()},
        "top_categories": {
            str(key): int(value)
            for key, value in df["normalized_category"].value_counts().head(20).to_dict().items()
        },
    }


def normalize_catalog(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> pd.DataFrame:
    """Run the end-to-end normalization pipeline and save the MVP catalog dataset."""

    raw_df = ingest_catalog(source_path=source_path, output_path=None)
    normalized_df = normalize_catalog_dataframe(raw_df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(output, index=False)

    summary = build_summary(normalized_df)
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return normalized_df
