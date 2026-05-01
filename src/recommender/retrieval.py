"""Retrieve candidate items from catalog and wardrobe sources."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CATALOG_PATH = Path("data/processed/catalog_items/catalog_items_mvp.csv")
ROLE_LIMITS = {
    "top": 20,
    "bottom": 15,
    "shoes": 12,
    "outerwear": 10,
}

FORMALITY_CATEGORY_WEIGHTS = {
    "formal": {
        "blazer": 10,
        "shirt": 8,
        "blouse": 8,
        "trousers": 8,
        "coat": 7,
        "boots": 4,
        "pumps": 5,
    },
    "smart_casual": {
        "blazer": 8,
        "shirt": 7,
        "blouse": 7,
        "sweater": 6,
        "trousers": 7,
        "skirt": 6,
        "jacket": 5,
        "boots": 4,
        "sneakers": 2,
    },
    "business": {
        "blazer": 10,
        "shirt": 8,
        "blouse": 8,
        "trousers": 9,
        "skirt": 7,
        "coat": 5,
        "pumps": 4,
        "boots": 4,
    },
    "casual": {
        "tshirt": 8,
        "hoodie": 8,
        "sweater": 6,
        "top": 6,
        "tank_top": 6,
        "shorts": 8,
        "sneakers": 7,
        "sandals": 6,
        "cardigan": 4,
    },
    "sporty": {
        "leggings_tights": 8,
        "outdoor_trousers": 8,
        "sneakers": 9,
        "hoodie": 5,
        "jacket": 4,
        "tshirt": 4,
    },
}

OCCASION_HINTS = {
    "work": {
        "preferred_section_themes": {"tailoring"},
        "avoid_section_themes": {"sport"},
    },
    "dinner": {
        "preferred_section_themes": {"tailoring", "trend", "contemporary_smart"},
        "avoid_section_themes": {"sport"},
    },
    "party": {
        "preferred_section_themes": {"trend", "divided_projects"},
        "avoid_section_themes": {"sport"},
    },
}

QUERY_TEXT_COLUMNS = [
    "display_name",
    "description",
    "normalized_category",
    "normalized_color",
    "normalized_pattern",
    "section_theme",
]

ROLE_CATEGORY_LIMITS = {
    "top": 3,
    "bottom": 3,
    "shoes": 2,
    "outerwear": 2,
}


@lru_cache(maxsize=1)
def load_catalog_items() -> pd.DataFrame:
    """Load the processed catalog once per process."""

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Processed catalog not found: {CATALOG_PATH}")
    return pd.read_csv(CATALOG_PATH)


def _score_query_term_overlap(row: pd.Series, search_terms: list[str]) -> int:
    if not search_terms:
        return 0
    haystack = " ".join(str(row.get(column, "")).lower() for column in QUERY_TEXT_COLUMNS)
    matched_terms = sum(1 for term in search_terms if term in haystack)
    return matched_terms * 2


def _score_category_preferences(row: pd.Series, preferred_categories: list[str]) -> int:
    if not preferred_categories:
        return 0
    category = str(row["normalized_category"]).lower()
    display_name = str(row["display_name"]).lower()
    if category in preferred_categories:
        return 10
    if any(keyword.replace("_", " ") in display_name for keyword in preferred_categories):
        return 6
    return 0


def _score_color_preferences(row: pd.Series, preferred_colors: list[str]) -> int:
    if not preferred_colors:
        return 0
    normalized_color = str(row["normalized_color"]).lower()
    color_detail = str(row["color_detail"]).lower()
    if normalized_color in preferred_colors:
        return 10
    if any(color in color_detail for color in preferred_colors):
        return 6
    return 0


def _score_formality_proxy(row: pd.Series, formality: str | None) -> int:
    if formality is None:
        return 0
    return FORMALITY_CATEGORY_WEIGHTS.get(formality, {}).get(str(row["normalized_category"]).lower(), 0)


def _score_occasion_proxy(row: pd.Series, occasion: str | None) -> int:
    if occasion is None:
        return 0
    hints = OCCASION_HINTS.get(occasion)
    if not hints:
        return 0
    section_theme = str(row["section_theme"]).lower()
    score = 0
    if section_theme in hints["preferred_section_themes"]:
        score += 4
    if section_theme in hints["avoid_section_themes"]:
        score -= 6
    return score


def _score_item(row: pd.Series, constraints: dict) -> int:
    # Start every candidate with a shared base score, then add query-specific boosts.
    score = 10
    score += _score_color_preferences(row, constraints["preferred_colors"])
    score += _score_category_preferences(row, constraints["preferred_categories"])
    score += _score_query_term_overlap(row, constraints["search_terms"])
    score += _score_formality_proxy(row, constraints["formality"])
    score += _score_occasion_proxy(row, constraints["occasion"])

    if constraints["formality"] in {"formal", "smart_casual", "business"} and str(row["section_theme"]).lower() == "sport":
        score -= 8

    return score


def _select_diverse_role_candidates(role_df: pd.DataFrame, role: str, limit: int) -> pd.DataFrame:
    """Keep strong items while preventing one category from dominating the role pool."""

    per_category_limit = ROLE_CATEGORY_LIMITS.get(role, 2)
    selected_rows = []
    category_counts: dict[str, int] = {}

    for _, row in role_df.iterrows():
        category = str(row["normalized_category"])
        current_count = category_counts.get(category, 0)

        if current_count >= per_category_limit:
            continue

        selected_rows.append(row.to_dict())
        category_counts[category] = current_count + 1

        if len(selected_rows) >= limit:
            break

    if len(selected_rows) < limit:
        selected_ids = {str(row["item_id"]) for row in selected_rows}
        for _, row in role_df.iterrows():
            if str(row["item_id"]) in selected_ids:
                continue
            selected_rows.append(row.to_dict())
            if len(selected_rows) >= limit:
                break

    return pd.DataFrame(selected_rows)


def retrieve_candidates_by_role(constraints: dict) -> dict[str, list[dict]]:
    """Retrieve and score item candidates for each required role."""

    catalog_df = load_catalog_items()
    target_group = constraints["target_group"]
    role_candidates: dict[str, list[dict]] = {}

    base_df = catalog_df[catalog_df["target_group"] == target_group].copy()

    for role in constraints["required_roles"]:
        role_df = base_df[base_df["recommendation_role"] == role].copy()
        if role_df.empty:
            role_candidates[role] = []
            continue

        # Score items within the same role so tops compete with tops, shoes with shoes, etc.
        role_df["candidate_score"] = role_df.apply(lambda row: _score_item(row, constraints), axis=1)
        role_df = role_df.sort_values(
            by=["candidate_score", "normalized_category", "display_name"],
            ascending=[False, True, True],
        )

        # Keep a small but category-diverse pool per role before outfit composition.
        limit = ROLE_LIMITS.get(role, 10)
        diversified_role_df = _select_diverse_role_candidates(role_df, role, limit)
        candidate_rows = diversified_role_df.to_dict(orient="records")
        role_candidates[role] = candidate_rows

    return role_candidates


def retrieve_candidate_items(constraints: dict) -> list[dict]:
    """Return a flattened list of the highest-scoring role-based item candidates."""

    role_candidates = retrieve_candidates_by_role(constraints)
    flattened: list[dict] = []
    for role in constraints["required_roles"]:
        flattened.extend(role_candidates.get(role, []))

    return sorted(flattened, key=lambda item: item.get("candidate_score", 0), reverse=True)
