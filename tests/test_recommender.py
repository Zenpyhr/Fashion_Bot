from pathlib import Path

import pandas as pd

from src.recommender.query_parser import parse_user_query
from src.recommender.retrieval import _resolve_catalog_path, _score_item, load_catalog_items
from src.shared.config import settings
from src.shared.schemas import RecommendationRequest


def test_recommendation_request_schema() -> None:
    payload = RecommendationRequest(user_query="Build me a spring outfit.")
    assert payload.user_query


def test_query_parser_extracts_basic_constraints() -> None:
    constraints = parse_user_query("Need a smart casual men's outfit in black.")

    assert constraints["target_group"] == "men"
    assert "black" in constraints["preferred_colors"]
    assert constraints["formality"] == "smart_casual"
    assert constraints["required_roles"] == ["top", "bottom", "shoes"]


def test_load_catalog_items_falls_back_from_legacy_data_processed_path(monkeypatch) -> None:
    fallback_csv = Path("data/recommender/processed/catalog_items/catalog_items_demo.csv")
    legacy_path = Path("data/processed/catalog_items/catalog_items_demo.csv")
    monkeypatch.setattr(settings, "catalog_items_csv", str(legacy_path))
    _resolve_catalog_path.cache_clear()
    load_catalog_items.cache_clear()

    try:
        resolved = _resolve_catalog_path(settings.catalog_items_csv)
        df = load_catalog_items()
    finally:
        _resolve_catalog_path.cache_clear()
        load_catalog_items.cache_clear()

    assert resolved == fallback_csv
    assert not df.empty


def test_score_item_handles_wardrobe_like_row_and_rewards_query_match() -> None:
    row = pd.Series(
        {
            "display_name": "Grey chunky athletic sneakers",
            "description": "casual everyday sneakers",
            "normalized_category": "sneakers",
            "normalized_color": "gray",
            "recommendation_role": "shoes",
            "section_theme": "casual",
        }
    )
    constraints = {
        "preferred_colors": ["gray"],
        "preferred_categories": ["sneakers"],
        "search_terms": ["casual", "sneakers"],
        "formality": "casual",
        "occasion": "casual",
        "intent_summer_lightweight": False,
        "intent_rainy_or_cold": False,
        "intent_polished": False,
        "intent_not_sporty": False,
        "semantic_query": "gray sneakers casual everyday",
        "raw_query": "Use my grey sneakers in a casual everyday outfit",
    }

    score = _score_item(row, constraints)

    assert score > 13
