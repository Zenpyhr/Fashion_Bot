"""Rank outfit candidates based on constraint match and compatibility."""

from __future__ import annotations

from itertools import product


def _outfit_similarity(left: dict, right: dict) -> int:
    """Higher value means the outfits are more repetitive relative to each other."""

    left_items = left["items"]
    right_items = right["items"]

    left_ids = {str(item.get("item_id")) for item in left_items}
    right_ids = {str(item.get("item_id")) for item in right_items}
    shared_ids = len(left_ids & right_ids)

    left_categories = [str(item.get("normalized_category")) for item in left_items]
    right_categories = [str(item.get("normalized_category")) for item in right_items]
    shared_categories = sum(1 for left_cat, right_cat in zip(left_categories, right_categories) if left_cat == right_cat)

    left_colors = [str(item.get("normalized_color")) for item in left_items]
    right_colors = [str(item.get("normalized_color")) for item in right_items]
    shared_colors = sum(1 for left_color, right_color in zip(left_colors, right_colors) if left_color == right_color)

    return shared_ids * 8 + shared_categories * 3 + shared_colors * 2


def _select_diverse_outfits(outfits: list[dict], limit: int = 10) -> list[dict]:
    """Greedily keep strong outfits while discouraging near-duplicates."""

    if not outfits:
        return []

    selected = [outfits[0]]
    remaining = outfits[1:]

    while remaining and len(selected) < limit:
        best_candidate = None
        best_adjusted_score = None

        selected_signatures = {
            tuple(str(item.get("normalized_category")) for item in outfit["items"])
            for outfit in selected
        }
        novel_structure_candidates = [
            candidate
            for candidate in remaining
            if tuple(str(item.get("normalized_category")) for item in candidate["items"]) not in selected_signatures
        ]
        candidate_pool = novel_structure_candidates or remaining

        for candidate in candidate_pool:
            max_similarity = max(_outfit_similarity(candidate, chosen) for chosen in selected)
            adjusted_score = candidate["score"] - max_similarity

            if tuple(str(item.get("normalized_category")) for item in candidate["items"]) in selected_signatures:
                adjusted_score -= 20

            if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining = [candidate for candidate in remaining if candidate is not best_candidate]

    return selected


def _color_cohesion_score(items: list[dict], preferred_colors: list[str]) -> int:
    colors = [str(item.get("normalized_color") or "") for item in items]
    if not colors:
        return 0

    score = 0
    if preferred_colors:
        matches = sum(1 for color in colors if color in preferred_colors)
        score += matches * 3

    unique_colors = {color for color in colors if color}
    if len(unique_colors) <= 2:
        score += 4
    elif len(unique_colors) == 3:
        score += 1
    return score


def _section_theme_penalty(items: list[dict], formality: str | None) -> int:
    if formality not in {"formal", "smart_casual", "business"}:
        return 0
    sport_count = sum(1 for item in items if str(item.get("section_theme", "")).lower() == "sport")
    return sport_count * -5


def rank_outfits(candidates_by_role: dict[str, list[dict]], constraints: dict) -> list[dict]:
    """Compose simple outfits from role-based candidates and rank them."""

    required_roles = constraints["required_roles"]
    if any(not candidates_by_role.get(role) for role in required_roles):
        return []

    trimmed_role_pools = {
        role: candidates_by_role[role][: 4 if role != "outerwear" else 3]
        for role in required_roles
    }

    # Build outfit combinations from the best few items in each role pool.
    ordered_pools = [trimmed_role_pools[role] for role in required_roles]
    outfits: list[dict] = []

    for combination in product(*ordered_pools):
        items = list(combination)
        # Start from the summed item quality, then adjust for outfit-level coherence.
        outfit_score = sum(int(item.get("candidate_score", 0)) for item in items)
        outfit_score += _color_cohesion_score(items, constraints["preferred_colors"])
        outfit_score += _section_theme_penalty(items, constraints["formality"])

        outfits.append(
            {
                "score": outfit_score,
                "items": items,
                "roles": required_roles,
            }
        )

    outfits.sort(key=lambda outfit: outfit["score"], reverse=True)
    return _select_diverse_outfits(outfits[:30], limit=10)
