"""Shared request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ItemRecord(BaseModel):
    item_id: str
    source_type: str = "catalog"
    product_code: str | None = None
    prod_name: str | None = None
    product_type_name: str | None = None
    product_group_name: str | None = None
    colour_group_name: str | None = None
    perceived_colour_master_name: str | None = None
    graphical_appearance_name: str | None = None
    department_name: str | None = None
    section_name: str | None = None
    garment_group_name: str | None = None
    detail_desc: str | None = None
    image_path: str | None = None
    recommendation_role: str | None = None
    normalized_category: str | None = None
    normalized_color: str | None = None
    normalized_pattern: str | None = None
    target_gender: str | None = None
    is_recommendable: bool = True
    is_user_owned: bool = False
    season: str | None = None
    formality: str | None = None
    style: str | None = None
    occasion: str | None = None
    vlm_enriched: bool = False
    vlm_confidence: float | None = None


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)


class QAResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    use_owned_only: bool = False
    user_id: str | None = None


class RecommendationResponse(BaseModel):
    parsed_constraints: dict[str, Any] = Field(default_factory=dict)
    outfits: list[dict[str, Any]] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
