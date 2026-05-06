"""Normalize wardrobe (VLM) metadata into canonical recommender values.

This module is intentionally strict: the recommender relies on closed sets for
role/category/theme. Unknown values must be quarantined or repaired, not silently
passed through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.recommender.canonical import ALLOWED_CATEGORIES, ALLOWED_ROLES, ALLOWED_SECTION_THEMES


def _norm(s: str | None) -> str:
    text = str(s or "").strip().lower()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


ROLE_MAP: dict[str, str] = {
    "shoe": "shoes",
    "shoes": "shoes",
    "footwear": "shoes",
    "sneaker": "shoes",
    "sneakers": "shoes",
    "boot": "shoes",
    "boots": "shoes",
    "outer": "outerwear",
    "outerwear": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "top": "top",
    "upper": "top",
    "upper body": "top",
    "bottom": "bottom",
    "pants": "bottom",
    "trousers": "bottom",
    "shorts": "bottom",
}

CATEGORY_MAP: dict[str, str] = {
    "t shirt": "tshirt",
    "t-shirt": "tshirt",
    "tee": "tshirt",
    "trainer": "sneakers",
    "trainers": "sneakers",
    "sneaker": "sneakers",
    "boot": "boots",
    "hooded sweatshirt": "hoodie",
}

THEME_MAP: dict[str, str] = {
    "active": "sport",
    "activewear": "sport",
    "athleisure": "sport",
    "sports": "sport",
    "street": "contemporary_street",
    "streetwear": "contemporary_street",
    "smart": "contemporary_smart",
}


def _keyword_guess_category(text: str) -> str | None:
    # Small, conservative keyword guesses when the VLM outputs a novel label.
    if "hoodie" in text:
        return "hoodie"
    if "cardigan" in text:
        return "cardigan"
    if "waistcoat" in text or "vest" in text:
        return "waistcoat"
    if "blazer" in text:
        return "blazer"
    if "coat" in text:
        return "coat"
    if "jacket" in text:
        return "jacket"
    if "sweater" in text or "knit" in text:
        return "sweater"
    if "polo" in text:
        return "polo_shirt"
    if "shirt" in text:
        return "shirt"
    if "tank" in text:
        return "tank_top"
    if "tshirt" in text or "t shirt" in text or "tee" in text:
        return "tshirt"
    if "trouser" in text or "pant" in text:
        return "trousers"
    if "short" in text:
        return "shorts"
    if "sneaker" in text or "trainer" in text:
        return "sneakers"
    if "boot" in text:
        return "boots"
    if "sandal" in text:
        return "sandals"
    return None


def _keyword_guess_role(category: str | None) -> str | None:
    if not category:
        return None
    if category in {"boots", "sneakers", "sandals"}:
        return "shoes"
    if category in {"coat", "jacket", "blazer", "cardigan"}:
        return "outerwear"
    if category in {"trousers", "shorts", "outdoor_trousers"}:
        return "bottom"
    if category in {"hoodie", "sweater", "shirt", "tshirt", "polo_shirt", "tank_top", "top", "waistcoat"}:
        return "top"
    return None


@dataclass(frozen=True)
class NormalizationResult:
    item: dict[str, Any]
    quarantine_reasons: list[str]


def normalize_wardrobe_item(vlm_item: dict[str, Any]) -> NormalizationResult:
    """Return a canonical item dict plus quarantine reasons (empty when clean)."""

    raw = dict(vlm_item or {})
    reasons: list[str] = []

    display = _norm(raw.get("display_name"))
    desc = _norm(raw.get("description"))
    type_name = _norm(raw.get("product_type_name"))
    combined_text = " ".join([display, type_name, desc]).strip()

    # Normalize category
    raw_cat = _norm(raw.get("normalized_category"))
    cat = CATEGORY_MAP.get(raw_cat, raw_cat)
    if cat not in ALLOWED_CATEGORIES:
        guess = _keyword_guess_category(combined_text)
        if guess and guess in ALLOWED_CATEGORIES:
            cat = guess
        else:
            reasons.append(f"unknown_category:{raw_cat or 'empty'}")
            cat = "unknown"

    # Normalize section_theme
    raw_theme = _norm(raw.get("section_theme"))
    theme = THEME_MAP.get(raw_theme, raw_theme)
    if theme not in ALLOWED_SECTION_THEMES:
        # Keep unknown themes rather than inventing; can be repaired later.
        reasons.append(f"unknown_theme:{raw_theme or 'empty'}")
        theme = "unknown"

    # Normalize role
    raw_role = _norm(raw.get("recommendation_role"))
    role = ROLE_MAP.get(raw_role, raw_role)
    if role not in ALLOWED_ROLES:
        role_guess = _keyword_guess_role(cat if cat != "unknown" else None)
        if role_guess:
            role = role_guess
        else:
            reasons.append(f"unknown_role:{raw_role or 'empty'}")
            role = "unknown"

    # Canonicalize source_type: we treat wardrobe items as separate from catalog items.
    source_type = "wardrobe"

    out = dict(raw)
    out["source_type"] = source_type
    out["normalized_category"] = cat
    out["section_theme"] = theme
    out["recommendation_role"] = role
    out["target_group"] = "men"  # consistent with current demo catalog

    return NormalizationResult(item=out, quarantine_reasons=reasons)

