"""Recommendation API routes."""

from fastapi import APIRouter

from src.recommender.outfits import build_outfits
from src.shared.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post("", response_model=RecommendationResponse)
def recommend_outfit(payload: RecommendationRequest) -> RecommendationResponse:
    result = build_outfits(payload.user_query, user_id=payload.user_id)
    return RecommendationResponse(
        parsed_constraints=result["parsed_constraints"],
        outfits=result["outfits"],
        explanations=[outfit["explanation"] for outfit in result["outfits"]],
        missing_items=result["missing_items"],
    )
