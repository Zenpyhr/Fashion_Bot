"""Turn user recommendation requests into structured constraints."""

from __future__ import annotations

import re

from src.integrations.openai_client import llm_parse_query, openai_is_configured
from src.shared.config import settings
from src.shared.constants import RECOMMENDATION_ROLES

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "give",
    "i",
    "in",
    "is",
    "look",
    "me",
    "need",
    "of",
    "on",
    "outfit",
    "please",
    "something",
    "the",
    "to",
    "want",
    "wear",
    "with",
}

COLOR_KEYWORDS = {
    "black": ["black"],
    "white": ["white", "off white"],
    "blue": ["blue", "navy", "dark blue", "light blue"],
    "grey": ["grey", "gray", "light grey", "dark grey"],
    "beige": ["beige", "cream"],
    "brown": ["brown", "mole"],
    "green": ["green", "khaki", "khaki green", "dark green"],
    "red": ["red", "burgundy"],
    "pink": ["pink"],
    "yellow": ["yellow"],
    "orange": ["orange"],
}

NEUTRAL_KEYWORDS = {"neutral", "neutrals"}
NEUTRAL_COLORS = ["black", "white", "grey", "beige", "brown"]

TARGET_GROUP_KEYWORDS = {
    "women": ["women", "woman", "womens", "women's", "lady", "ladies", "female", "girl"],
    "men": ["men", "man", "mens", "men's", "male", "gentleman", "guy"],
}

ROLE_TRIGGER_KEYWORDS = {
    "outerwear": ["jacket", "coat", "outerwear", "blazer", "cardigan"],
    "shoes": ["shoe", "shoes", "sneaker", "sneakers", "boots", "sandals", "heels", "pumps"],
}

CATEGORY_KEYWORDS = {
    "blazer": ["blazer"],
    "blouse": ["blouse"],
    "boots": ["boots", "boot"],
    "cardigan": ["cardigan"],
    "coat": ["coat"],
    "hoodie": ["hoodie"],
    "jacket": ["jacket"],
    "leggings_tights": ["leggings", "tights"],
    "polo_shirt": ["polo"],
    "sandals": ["sandals", "sandal"],
    "shirt": ["shirt"],
    "shorts": ["shorts"],
    "skirt": ["skirt"],
    "sneakers": ["sneakers", "sneaker"],
    "sweater": ["sweater", "knitwear", "knit"],
    "tank_top": ["tank", "tank top", "vest top"],
    "top": ["top"],
    "trousers": ["trousers", "pants", "pant"],
    "tshirt": ["t-shirt", "tshirt", "tee"],
}

FORMALITY_KEYWORDS = {
    "formal": ["formal", "elegant", "dressy"],
    "smart_casual": ["smart casual", "smart-casual", "business casual"],
    "business": ["business", "office", "work", "professional"],
    "casual": ["casual", "everyday", "daily", "relaxed"],
    "sporty": ["sporty", "athletic", "running", "gym", "workout", "activewear"],
}

OCCASION_KEYWORDS = {
    "work": ["work", "office", "meeting"],
    "dinner": ["dinner"],
    "party": ["party"],
    "date_night": ["date", "date night"],
    "travel": ["travel", "airport"],
    "casual": ["casual", "everyday", "daily"],
}

WEATHER_OUTERWEAR_KEYWORDS = ["cold", "chilly", "rain", "rainy", "winter", "fall", "autumn"]


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _deterministic_parse_user_query(user_query: str) -> dict:
    """Parse a free-text recommendation query into structured constraints."""

    normalized_query = user_query.strip().lower()
    tokens = re.findall(r"[a-z0-9']+", normalized_query)
    search_terms = [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    target_group = None
    for group, keywords in TARGET_GROUP_KEYWORDS.items():
        if _contains_keyword(normalized_query, keywords):
            target_group = group
            break
    if target_group is None:
        # The current demo catalog only contains men's items, so default to men
        # to avoid empty retrieval when the user does not specify a target group.
        target_group = "men"

    preferred_colors: list[str] = []
    if any(keyword in normalized_query for keyword in NEUTRAL_KEYWORDS):
        preferred_colors.extend(NEUTRAL_COLORS)
    for color, keywords in COLOR_KEYWORDS.items():
        if _contains_keyword(normalized_query, keywords) and color not in preferred_colors:
            preferred_colors.append(color)

    preferred_categories = [
        category
        for category, keywords in CATEGORY_KEYWORDS.items()
        if _contains_keyword(normalized_query, keywords)
    ]

    include_outerwear = any(
        _contains_keyword(normalized_query, keywords)
        for role, keywords in ROLE_TRIGGER_KEYWORDS.items()
        if role == "outerwear"
    ) or any(keyword in normalized_query for keyword in WEATHER_OUTERWEAR_KEYWORDS)

    # The MVP always tries to build a complete base outfit first.
    required_roles = ["top", "bottom", "shoes"]
    if include_outerwear:
        required_roles.append("outerwear")

    requested_roles = [
        role
        for role, keywords in ROLE_TRIGGER_KEYWORDS.items()
        if _contains_keyword(normalized_query, keywords)
    ]

    formality = None
    for label, keywords in FORMALITY_KEYWORDS.items():
        if _contains_keyword(normalized_query, keywords):
            formality = label
            break

    occasion = None
    for label, keywords in OCCASION_KEYWORDS.items():
        if _contains_keyword(normalized_query, keywords):
            occasion = label
            break

    return {
        "raw_query": user_query,
        "target_group": target_group,
        "required_roles": required_roles,
        "requested_roles": requested_roles,
        "preferred_colors": preferred_colors,
        "preferred_categories": preferred_categories,
        "formality": formality,
        "occasion": occasion,
        "search_terms": search_terms,
        "available_roles": list(RECOMMENDATION_ROLES),
    }


def _merge_llm_constraints(base: dict, overrides: dict | None) -> dict:
    if not overrides:
        return base

    merged = dict(base)
    for key in (
        "target_group",
        "required_roles",
        "requested_roles",
        "preferred_colors",
        "preferred_categories",
        "formality",
        "occasion",
        "search_terms",
    ):
        if key in overrides and overrides[key] not in (None, "", []):
            merged[key] = overrides[key]
    return merged


def parse_user_query(user_query: str) -> dict:
    """Parse a recommendation query with deterministic logic plus optional OpenAI refinement."""

    deterministic_constraints = _deterministic_parse_user_query(user_query)
    deterministic_constraints["parser_source"] = "deterministic"

    if openai_is_configured() and settings.enable_openai_query_parser:
        llm_constraints = llm_parse_query(user_query, deterministic_constraints)
        merged = _merge_llm_constraints(deterministic_constraints, llm_constraints)
        if llm_constraints:
            merged["parser_source"] = "openai"
        return merged

    return deterministic_constraints
