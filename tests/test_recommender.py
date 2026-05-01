from src.shared.schemas import RecommendationRequest


def test_recommendation_request_schema() -> None:
    payload = RecommendationRequest(user_query="Build me a spring outfit.")
    assert payload.user_query
