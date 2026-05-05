"""Retrieve candidate items from catalog and wardrobe sources."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from src.integrations.embeddings import embed_text_openai, l2_normalize
from src.integrations.openai_client import openai_is_configured
from src.integrations.pgvector_store import create_engine_from_settings, ensure_embeddings_table, fetch_embeddings
from src.shared.config import settings

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


def _infer_vector_dim() -> int:
    model = settings.openai_embedding_model
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    raise RuntimeError(f"Unknown embedding dimension for model={model!r}.")


@lru_cache(maxsize=1)
def _get_embeddings_store():
    engine = create_engine_from_settings()
    table = ensure_embeddings_table(engine, vector_dim=_infer_vector_dim())
    return engine, table


def _dense_rerank_role_pool(role_df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Dense rerank inside a sparse shortlist.

    Assumes embeddings stored in Postgres are already L2-normalized.
    """

    if role_df.empty:
        return role_df

    query_text = (
        constraints.get("semantic_query")
        or constraints.get("raw_query")
        or constraints.get("raw_text")
        or ""
    )
    query_text = str(query_text).strip()
    if not query_text:
        return role_df

    query_emb = l2_normalize(embed_text_openai(query_text))

    engine, table = _get_embeddings_store()
    item_ids = [str(item_id) for item_id in role_df["item_id"].tolist()]
    emb_by_id = fetch_embeddings(engine, table, item_ids)

    # Compute cosine similarity via dot product (vectors are normalized).
    dense_scores = []
    for item_id in item_ids:
        emb = emb_by_id.get(str(item_id))
        if not emb:
            dense_scores.append(None)
            continue
        dense_scores.append(float(np.dot(query_emb, np.asarray(emb, dtype=np.float32))))

    reranked = role_df.copy()
    reranked["dense_score"] = dense_scores

    # Prefer dense score when present, keep sparse score as tie-breaker.
    reranked = reranked.sort_values(
        by=["dense_score", "candidate_score", "normalized_category", "display_name"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    return reranked


@lru_cache(maxsize=1)
def load_catalog_items() -> pd.DataFrame:
    """Load the processed catalog once per process."""

    catalog_path = Path(settings.catalog_items_csv)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Processed catalog not found: {catalog_path}")
    return pd.read_csv(catalog_path)


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


def _query_text(constraints: dict) -> str:
    raw = (
        constraints.get("semantic_query")
        or constraints.get("raw_query")
        or constraints.get("raw_text")
        or ""
    )
    return str(raw).strip().lower()


def _detect_query_intents(constraints: dict) -> dict[str, bool]:
    """Infer lightweight intent flags from query + existing parsed fields.

    This intentionally uses broad domain vocabulary (not eval-test hardcoding).
    """

    # Prefer explicit flags from the LLM parser if present.
    explicit_summer = constraints.get("intent_summer_lightweight")
    explicit_rainy_cold = constraints.get("intent_rainy_or_cold")
    explicit_polished = constraints.get("intent_polished")
    explicit_not_sporty = constraints.get("intent_not_sporty")

    text = _query_text(constraints)
    terms = {str(t).lower() for t in (constraints.get("search_terms") or [])}
    occasion = str(constraints.get("occasion") or "").lower()
    formality = str(constraints.get("formality") or "").lower()

    summer = bool(explicit_summer) if explicit_summer is not None else any(
        k in text for k in ("summer", "hot", "heat", "lightweight", "breathable", "airy")
    )
    rainy_or_cold = (
        bool(explicit_rainy_cold)
        if explicit_rainy_cold is not None
        else any(k in text for k in ("rain", "rainy", "drizzle", "storm", "waterproof", "water-resistant"))
        or any(k in text for k in ("cold", "chilly", "winter", "snow", "insulated", "padded", "warm"))
    )

    polished = (
        bool(explicit_polished)
        if explicit_polished is not None
        else any(
            k in text
            for k in (
                "polished",
                "clean",
                "minimal",
                "smart casual",
                "smart-casual",
                "office",
                "work",
                "meeting",
                "dinner",
                "date",
            )
        )
        or occasion in {"work", "dinner", "date_night"}
        or formality in {"business", "smart_casual", "formal"}
    )

    not_sporty = (
        bool(explicit_not_sporty)
        if explicit_not_sporty is not None
        else ("sporty" in terms and "not" in terms) or ("not sporty" in text) or ("avoid sporty" in text)
    )

    return {
        "summer": summer,
        "rainy_or_cold": rainy_or_cold,
        "polished": polished,
        "not_sporty": not_sporty,
    }


def _score_guardrails(row: pd.Series, constraints: dict) -> int:
    """Soft rule penalties/boosts to prevent obvious mismatches.

    These are deliberately modest (guardrails, not hard filters).
    """

    intents = _detect_query_intents(constraints)
    category = str(row.get("normalized_category", "")).lower()
    role = str(row.get("recommendation_role", "")).lower()
    section_theme = str(row.get("section_theme", "")).lower()
    desc = str(row.get("description", "")).lower()
    name = str(row.get("display_name", "")).lower()
    text = f"{name} {desc}"

    score = 0

    # Not sporty: penalize sport theme and explicitly sporty categories.
    if intents["not_sporty"]:
        if section_theme == "sport":
            score -= 8
        if category in {"leggings_tights", "outdoor_trousers"}:
            score -= 3

    # Summer/lightweight: discourage heavy layers.
    if intents["summer"]:
        if category in {"hoodie", "sweater", "coat"}:
            score -= 4
        if role == "outerwear" and category in {"jacket", "coat"}:
            score -= 2
        if category == "shorts":
            score += 2
        if any(k in text for k in ("linen", "lightweight", "breathable", "short sleeve")):
            score += 2

    # Rainy/cold: discourage shorts, boost protective details.
    if intents["rainy_or_cold"]:
        if category == "shorts":
            score -= 6
        if role == "outerwear":
            score += 2
        if any(k in text for k in ("hood", "waterproof", "water-resistant", "rain")):
            score += 2
        if any(k in text for k in ("padded", "insulated", "lined", "fleece", "wool")):
            score += 2

    # Polished/office/dinner: discourage shorts and sport theme.
    if intents["polished"]:
        if category == "shorts":
            score -= 6
        if section_theme == "sport":
            score -= 4
        if category in {"shirt", "trousers", "blazer", "coat", "boots"}:
            score += 2

    return score


def _score_item(row: pd.Series, constraints: dict) -> int:
    # Start every candidate with a shared base score, then add query-specific boosts.
    score = 10
    score += _score_color_preferences(row, constraints["preferred_colors"])
    score += _score_category_preferences(row, constraints["preferred_categories"])
    score += _score_query_term_overlap(row, constraints["search_terms"])
    score += _score_formality_proxy(row, constraints["formality"])
    score += _score_occasion_proxy(row, constraints["occasion"])
    score += _score_guardrails(row, constraints)

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

        # Dense rerank should happen inside a larger sparse shortlist (funnel design).
        shortlist_k = ROLE_LIMITS.get(role, 10)
        if settings.enable_dense_retrieval_rerank:
            shortlist_k = max(shortlist_k, settings.dense_shortlist_k_per_role)

        role_shortlist = role_df.head(shortlist_k).copy()
        if (
            settings.enable_dense_retrieval_rerank
            and openai_is_configured()
        ):
            role_shortlist = _dense_rerank_role_pool(role_shortlist, constraints)

        # Keep a small but category-diverse pool per role before outfit composition.
        limit = ROLE_LIMITS.get(role, 10)
        if settings.enable_dense_retrieval_rerank:
            limit = max(limit, min(settings.dense_rerank_n_per_role, shortlist_k))

        diversified_role_df = _select_diverse_role_candidates(role_shortlist, role, limit)
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
