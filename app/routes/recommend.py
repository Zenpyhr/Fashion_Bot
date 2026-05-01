"""Recommendation API routes."""

from fastapi import APIRouter

from src.shared.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post("", response_model=RecommendationResponse)
def recommend_outfit(payload: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(
        parsed_constraints={"raw_query": payload.user_query},
        outfits=[],
        explanations=["Recommendation pipeline not implemented yet."],
        missing_items=[],
    )
