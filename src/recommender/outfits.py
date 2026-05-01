"""Assemble outfit responses from ranked item candidates."""

from __future__ import annotations

from src.integrations.openai_client import llm_rerank_outfits, openai_is_configured
from src.recommender.query_parser import parse_user_query
from src.recommender.ranker import rank_outfits
from src.recommender.retrieval import retrieve_candidates_by_role
from src.shared.config import settings


def _format_item_summary(item: dict) -> dict:
    return {
        "item_id": item["item_id"],
        "display_name": item["display_name"],
        "recommendation_role": item["recommendation_role"],
        "normalized_category": item["normalized_category"],
        "normalized_color": item["normalized_color"],
        "section_theme": item["section_theme"],
        "score": item.get("candidate_score"),
    }


def _build_explanation(outfit: dict, constraints: dict) -> str:
    colors = [item.get("normalized_color") for item in outfit["items"] if item.get("normalized_color")]
    unique_colors = sorted({str(color) for color in colors})
    categories = [str(item.get("normalized_category")) for item in outfit["items"]]

    explanation_parts = [
        f"This outfit covers the required roles for a {constraints['target_group']} look.",
        f"It combines {' + '.join(categories)}.",
    ]

    if constraints["preferred_colors"]:
        explanation_parts.append(
            f"It leans into your requested color direction: {', '.join(constraints['preferred_colors'])}."
        )
    elif unique_colors:
        explanation_parts.append(f"The palette stays around {', '.join(unique_colors[:3])}.")

    if constraints["formality"]:
        explanation_parts.append(f"It also reflects a {constraints['formality']} tone using metadata-based heuristics.")

    return " ".join(explanation_parts)


def _outfit_signature(outfit: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(item.get("normalized_category")),
            str(item.get("normalized_color")),
        )
        for item in outfit["items"]
    )


def _outfit_similarity(left: dict, right: dict) -> int:
    left_signature = _outfit_signature(left)
    right_signature = _outfit_signature(right)

    same_slots = sum(1 for left_part, right_part in zip(left_signature, right_signature) if left_part == right_part)
    same_categories = sum(1 for (left_cat, _), (right_cat, _) in zip(left_signature, right_signature) if left_cat == right_cat)
    same_colors = sum(1 for (_, left_color), (_, right_color) in zip(left_signature, right_signature) if left_color == right_color)

    return same_slots * 6 + same_categories * 3 + same_colors * 2


def _select_top_diverse_outfits(ranked_outfits: list[dict], limit: int = 3) -> list[dict]:
    """Greedily choose final outfits while discouraging near-duplicates."""

    if not ranked_outfits:
        return []

    selected = [ranked_outfits[0]]
    remaining = ranked_outfits[1:]

    while remaining and len(selected) < limit:
        best_candidate = None
        best_adjusted_score = None

        for candidate in remaining:
            similarity_penalty = max(_outfit_similarity(candidate, chosen) for chosen in selected)
            adjusted_score = int(candidate.get("score", 0)) - similarity_penalty

            if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining = [candidate for candidate in remaining if candidate is not best_candidate]

    return selected


def _prepare_outfits_for_llm(ranked_outfits: list[dict]) -> list[dict]:
    prepared = []
    for index, outfit in enumerate(ranked_outfits[:8], start=1):
        prepared.append(
            {
                "outfit_id": f"outfit_{index}",
                "current_score": outfit["score"],
                "signature": " + ".join(
                    f"{item.get('normalized_category')}:{item.get('normalized_color')}"
                    for item in outfit["items"]
                ),
                "items": [
                    {
                        "display_name": item.get("display_name"),
                        "role": item.get("recommendation_role"),
                        "category": item.get("normalized_category"),
                        "color": item.get("normalized_color"),
                        "section_theme": item.get("section_theme"),
                    }
                    for item in outfit["items"]
                ],
            }
        )
    return prepared


def _prepare_selected_outfits_for_llm(outfits: list[dict]) -> list[dict]:
    prepared = []
    for index, outfit in enumerate(outfits, start=1):
        prepared.append(
            {
                "outfit_id": f"selected_outfit_{index}",
                "current_score": outfit["score"],
                "signature": " + ".join(
                    f"{item.get('normalized_category')}:{item.get('normalized_color')}"
                    for item in outfit["items"]
                ),
                "items": [
                    {
                        "display_name": item.get("display_name"),
                        "role": item.get("recommendation_role"),
                        "category": item.get("normalized_category"),
                        "color": item.get("normalized_color"),
                        "section_theme": item.get("section_theme"),
                    }
                    for item in outfit["items"]
                ],
            }
        )
    return prepared


def _apply_llm_reranking(user_query: str, constraints: dict, ranked_outfits: list[dict]) -> list[dict]:
    if not ranked_outfits:
        return ranked_outfits

    llm_payload = _prepare_outfits_for_llm(ranked_outfits)
    llm_result = llm_rerank_outfits(user_query, constraints, llm_payload)
    if not llm_result:
        return ranked_outfits

    ranked_ids = llm_result.get("ranked_outfit_ids", [])
    explanations = llm_result.get("explanations", {})
    id_to_outfit = {f"outfit_{idx}": outfit for idx, outfit in enumerate(ranked_outfits[:8], start=1)}

    reranked: list[dict] = []
    seen = set()
    for outfit_id in ranked_ids:
        outfit = id_to_outfit.get(outfit_id)
        if outfit is None:
            continue
        updated = dict(outfit)
        if outfit_id in explanations:
            updated["llm_explanation"] = explanations[outfit_id]
        reranked.append(updated)
        seen.add(outfit_id)

    for outfit_id, outfit in id_to_outfit.items():
        if outfit_id not in seen:
            reranked.append(outfit)

    return reranked + ranked_outfits[8:]


def _apply_llm_explanations_to_selected_outfits(
    user_query: str,
    constraints: dict,
    selected_outfits: list[dict],
) -> list[dict]:
    """Ensure the final returned outfits all receive consistent LLM explanations."""

    if not selected_outfits:
        return selected_outfits

    llm_payload = _prepare_selected_outfits_for_llm(selected_outfits)
    llm_result = llm_rerank_outfits(user_query, constraints, llm_payload)
    if not llm_result:
        return selected_outfits

    explanations = llm_result.get("explanations", {})
    updated_outfits = []
    for index, outfit in enumerate(selected_outfits, start=1):
        updated = dict(outfit)
        outfit_id = f"selected_outfit_{index}"
        if outfit_id in explanations:
            updated["llm_explanation"] = explanations[outfit_id]
        updated_outfits.append(updated)

    return updated_outfits


def build_outfits(user_query: str) -> dict:
    """Parse query, retrieve role candidates, and return the top outfit suggestions."""

    # Orchestrator for the MVP recommender:
    # 1) understand the query
    # 2) retrieve item pools per role
    # 3) combine and rank outfits
    constraints = parse_user_query(user_query)
    candidates_by_role = retrieve_candidates_by_role(constraints)
    ranked_outfits = rank_outfits(candidates_by_role, constraints)
    reranker_source = "deterministic"
    if openai_is_configured() and settings.enable_openai_reranker:
        reranked_outfits = _apply_llm_reranking(user_query, constraints, ranked_outfits)
        if reranked_outfits is not ranked_outfits:
            reranker_source = "openai"
        ranked_outfits = reranked_outfits

    final_outfits = _select_top_diverse_outfits(ranked_outfits, limit=3)
    if openai_is_configured() and settings.enable_openai_reranker:
        final_outfits = _apply_llm_explanations_to_selected_outfits(user_query, constraints, final_outfits)

    formatted_outfits = [
        {
            "score": outfit["score"],
            "items": [_format_item_summary(item) for item in outfit["items"]],
            "explanation": outfit.get("llm_explanation") or _build_explanation(outfit, constraints),
        }
        for outfit in final_outfits
    ]

    missing_items = [role for role in constraints["required_roles"] if not candidates_by_role.get(role)]

    return {
        "parsed_constraints": constraints,
        "llm_status": {
            "query_parser": constraints.get("parser_source", "deterministic"),
            "reranker": reranker_source,
        },
        "candidate_pools": {
            role: [_format_item_summary(item) for item in items[:5]]
            for role, items in candidates_by_role.items()
        },
        "outfits": formatted_outfits,
        "missing_items": missing_items,
    }
