from src.recommender.query_parser import parse_user_query
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
