"""Assemble outfit responses from ranked item candidates."""

from __future__ import annotations

import logging

from src.integrations.openai_client import llm_compose_outfits, llm_rerank_outfits, openai_is_configured
from src.recommender.query_parser import parse_user_query
from src.recommender.ranker import rank_outfits, score_outfit_items
from src.recommender.retrieval import retrieve_candidates_by_role
from src.shared.config import settings


def _image_url_from_path(image_path: str | None) -> str | None:
    if not image_path:
        return None

    normalized_path = image_path.replace("\\", "/")
    # Support relative paths plus absolute paths on Windows/macOS. Keep legacy prefixes so
    # older DB rows and eval artifacts still map to URLs after the data/ layout move.
    demo_prefixes = (
        "data/recommender/processed/demo_images/",
        "data/processed/demo_images/",
    )
    for demo_prefix in demo_prefixes:
        if normalized_path.startswith(demo_prefix):
            return f"/demo_images/{normalized_path.removeprefix(demo_prefix)}"
        if f"/{demo_prefix}" in normalized_path:
            return f"/demo_images/{normalized_path.split(f'/{demo_prefix}', 1)[1]}"

    wardrobe_prefixes = (
        "data/recommender/user_wardrobe/",
        "data/user_wardrobe/",
    )
    for wardrobe_prefix in wardrobe_prefixes:
        if normalized_path.startswith(wardrobe_prefix):
            return f"/user_wardrobe/{normalized_path.removeprefix(wardrobe_prefix)}"
        if f"/{wardrobe_prefix}" in normalized_path:
            return f"/user_wardrobe/{normalized_path.split(f'/{wardrobe_prefix}', 1)[1]}"
    return None


def _format_item_summary(item: dict) -> dict:
    image_path = item.get("image_path")
    return {
        "item_id": item["item_id"],
        "source_type": item.get("source_type") or "catalog",
        "display_name": item["display_name"],
        "recommendation_role": item["recommendation_role"],
        "normalized_category": item["normalized_category"],
        "normalized_color": item["normalized_color"],
        "section_theme": item["section_theme"],
        "image_path": image_path,
        "image_url": _image_url_from_path(image_path),
        "image_relative_path": item.get("image_relative_path"),
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


def _outfit_category_signature(outfit: dict) -> tuple[str, ...]:
    return tuple(str(item.get("normalized_category")) for item in outfit["items"])


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
    used_item_ids = {str(item.get("item_id")) for item in ranked_outfits[0].get("items", []) if item.get("item_id") is not None}
    selected_category_signatures = {_outfit_category_signature(ranked_outfits[0])}

    while remaining and len(selected) < limit:
        best_candidate = None
        best_adjusted_score = None

        # Prefer outfits with a new role-category structure first. Only reuse the same
        # structure when the shortlist truly offers no other viable direction.
        novel_structure_candidates = [
            candidate
            for candidate in remaining
            if _outfit_category_signature(candidate) not in selected_category_signatures
        ]

        def _candidate_item_ids(candidate: dict) -> set[str]:
            return {
                str(item.get("item_id"))
                for item in candidate.get("items", [])
                if item.get("item_id") is not None
            }

        # Consider ALL remaining outfits for disjoint item_ids. Previously we only scanned
        # novel_structure_candidates when that list was non-empty, which missed disjoint
        # outfits that shared the same category signature as an already-picked outfit.
        non_overlapping = [c for c in remaining if _candidate_item_ids(c).isdisjoint(used_item_ids)]

        if non_overlapping:
            novel_disjoint = [
                c for c in non_overlapping if _outfit_category_signature(c) not in selected_category_signatures
            ]
            candidate_pool = novel_disjoint or non_overlapping
        else:
            candidate_pool = novel_structure_candidates or remaining

        for candidate in candidate_pool:
            similarity_penalty = max(_outfit_similarity(candidate, chosen) for chosen in selected)
            adjusted_score = int(candidate.get("score", 0)) - similarity_penalty

            if _outfit_category_signature(candidate) in selected_category_signatures:
                adjusted_score -= 40

            if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        used_item_ids |= {
            str(item.get("item_id"))
            for item in best_candidate.get("items", [])
            if item.get("item_id") is not None
        }
        selected_category_signatures.add(_outfit_category_signature(best_candidate))
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
                        "item_id": item.get("item_id"),
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
                        "item_id": item.get("item_id"),
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


def _compact_pool_item_for_llm(item: dict) -> dict:
    return {
        "item_id": item.get("item_id"),
        "display_name": item.get("display_name"),
        "recommendation_role": item.get("recommendation_role"),
        "normalized_category": item.get("normalized_category"),
        "normalized_color": item.get("normalized_color"),
        "section_theme": item.get("section_theme"),
        "candidate_score": item.get("candidate_score"),
    }


def _trim_pools_for_combo_llm(candidates_by_role: dict[str, list[dict]]) -> dict[str, list[dict]]:
    cap = max(4, settings.combo_composer_max_items_per_role_for_llm)
    return {role: items[:cap] for role, items in candidates_by_role.items()}


def _build_role_item_lookup(pools: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """role -> item_id -> full item dict (first wins)."""

    out: dict[str, dict[str, dict]] = {}
    for role, items in pools.items():
        id_map: dict[str, dict] = {}
        for item in items:
            iid = item.get("item_id")
            if iid is None:
                continue
            sid = str(iid)
            if sid not in id_map:
                id_map[sid] = item
        out[role] = id_map
    return out


def _outfits_from_llm_compose(
    llm_result: dict,
    constraints: dict,
    lookup: dict[str, dict[str, dict]],
) -> list[dict] | None:
    """Turn validated llm_compose_outfits JSON into outfit dicts, or None if invalid."""

    raw = llm_result.get("outfits")
    if not isinstance(raw, list) or len(raw) != 3:
        return None

    required_roles = constraints["required_roles"]
    used_ids: set[str] = set()
    outfits: list[dict] = []

    for entry in raw:
        if not isinstance(entry, dict):
            return None
        by_role = entry.get("items_by_role")
        if not isinstance(by_role, dict):
            return None

        items: list[dict] = []
        for role in required_roles:
            rid = by_role.get(role)
            if rid is None:
                return None
            sid = str(rid)
            item = lookup.get(role, {}).get(sid)
            if item is None:
                return None
            if sid in used_ids:
                return None
            used_ids.add(sid)
            items.append(item)

        explanation = entry.get("explanation")
        ex = str(explanation).strip() if explanation is not None else ""

        outfit = {
            "score": score_outfit_items(items, constraints),
            "items": items,
            "roles": required_roles,
            "llm_explanation": ex if ex else None,
        }
        outfits.append(outfit)

    return outfits


def _try_llm_compose_outfits(
    user_query: str,
    constraints: dict,
    candidates_by_role: dict[str, list[dict]],
) -> list[dict] | None:
    """Grounded LLM combo builder; returns None on missing config, API failure, or invalid JSON."""

    if not settings.enable_openai_combo_composer:
        return None
    if not openai_is_configured():
        return None

    required_roles = constraints["required_roles"]
    if not required_roles or any(not candidates_by_role.get(role) for role in required_roles):
        return None

    pools = _trim_pools_for_combo_llm(candidates_by_role)
    compact_pools = {
        role: [_compact_pool_item_for_llm(it) for it in pools.get(role, [])] for role in required_roles
    }
    llm_result = llm_compose_outfits(user_query, constraints, compact_pools)
    if not llm_result:
        return None

    lookup = _build_role_item_lookup(pools)
    return _outfits_from_llm_compose(llm_result, constraints, lookup)


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


def build_outfits(user_query: str, *, user_id: str | None = None) -> dict:
    """Parse query, retrieve role candidates, and return the top outfit suggestions."""

    # Orchestrator for the MVP recommender:
    # 1) understand the query
    # 2) retrieve item pools per role
    # 3) combine and rank outfits (deterministic product, or grounded LLM compose when enabled)
    constraints = parse_user_query(user_query)
    if user_id:
        constraints["user_id"] = user_id
    candidates_by_role = retrieve_candidates_by_role(constraints)

    combo_builder_source = "deterministic"
    ranked_outfits = _try_llm_compose_outfits(user_query, constraints, candidates_by_role)
    if ranked_outfits:
        combo_builder_source = "openai"
    else:
        ranked_outfits = rank_outfits(candidates_by_role, constraints)

    reranker_source = "deterministic"
    if combo_builder_source != "openai" and openai_is_configured() and settings.enable_openai_reranker:
        reranked_outfits = _apply_llm_reranking(user_query, constraints, ranked_outfits)
        if reranked_outfits is not ranked_outfits:
            reranker_source = "openai"
        ranked_outfits = reranked_outfits

    if combo_builder_source == "openai":
        final_outfits = ranked_outfits[:3]
        reranker_source = "skipped"
    else:
        final_outfits = _select_top_diverse_outfits(ranked_outfits, limit=3)

    need_explanations = not (
        combo_builder_source == "openai" and all(o.get("llm_explanation") for o in final_outfits)
    )
    if need_explanations and openai_is_configured() and settings.enable_openai_reranker:
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
            "combo_builder": combo_builder_source,
            "reranker": reranker_source,
        },
        "candidate_pools": {
            role: [_format_item_summary(item) for item in items[:5]]
            for role, items in candidates_by_role.items()
        },
        "outfits": formatted_outfits,
        "missing_items": missing_items,
    }


