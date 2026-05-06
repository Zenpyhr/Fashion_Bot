"""Canonical vocabularies for the recommender.

Wardrobe tagging (VLM) must be normalized into these values so retrieval/ranking
logic behaves consistently with the catalog.
"""

from __future__ import annotations

from typing import Final


ALLOWED_ROLES: Final[set[str]] = {"top", "bottom", "shoes", "outerwear"}

# Derived from `data/recommender/processed/catalog_items/catalog_items_demo.csv`.
ALLOWED_CATEGORIES: Final[set[str]] = {
    "blazer",
    "boots",
    "cardigan",
    "coat",
    "hoodie",
    "jacket",
    "outdoor_trousers",
    "polo_shirt",
    "sandals",
    "shirt",
    "shorts",
    "sneakers",
    "sweater",
    "tank_top",
    "top",
    "trousers",
    "tshirt",
    "waistcoat",
}

# Derived from `data/recommender/processed/catalog_items/catalog_items_demo.csv`.
ALLOWED_SECTION_THEMES: Final[set[str]] = {
    "basics",
    "casual",
    "contemporary_smart",
    "contemporary_street",
    "denim_men",
    "men_edition",
    "men_other",
    "men_other_2",
    "men_project",
    "men_shoes",
    "men_underwear",
    "mens_outerwear",
    "sport",
    "tailoring",
}

