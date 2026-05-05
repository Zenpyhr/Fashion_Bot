"""OpenAI integration helpers for parsing and reranking."""

from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime fallback for environments without the package yet.
    OpenAI = None

from src.shared.config import settings


def openai_is_configured() -> bool:
    return OpenAI is not None and bool(settings.openai_api_key.strip())


def create_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed. Run `pip install -r requirements.txt`.")
    return OpenAI(api_key=settings.openai_api_key)


def _call_openai_json(
    *,
    model: str,
    instructions: str,
    payload: dict[str, Any],
    max_output_tokens: int = 500,
) -> dict[str, Any] | None:
    """Request JSON output from the Responses API and parse it."""

    if not openai_is_configured():
        return None

    client = create_openai_client()

    # Some models/endpoints may not support the `reasoning` field. We first try with it,
    # then retry without reasoning on failure.
    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": settings.openai_reasoning_effort},
            instructions=instructions,
            input=json.dumps(payload),
            max_output_tokens=max_output_tokens,
        )
    except Exception:
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=json.dumps(payload),
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            # Fall back cleanly when the API is unavailable, blocked by the environment, or misconfigured.
            return None

    raw_text = (response.output_text or "").strip()
    if not raw_text:
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def llm_parse_query(user_query: str, deterministic_constraints: dict[str, Any]) -> dict[str, Any] | None:
    """Use OpenAI to refine the query constraints into structured JSON."""

    instructions = (
        "You are a fashion recommendation query parser for a MEN'S-only catalog (target_group must be 'men'). "
        "Return valid JSON only. Do not include markdown. "
        "Keep the schema shape aligned with the provided deterministic constraints.\n\n"
        "Allowed values:\n"
        "- target_group: men\n"
        "- required_roles/requested_roles: top, bottom, shoes, outerwear\n"
        "- formality: casual, smart_casual, business, formal, sporty, or null\n"
        "- occasion: work, dinner, party, date_night, travel, casual, or null\n\n"
        "Negation rule:\n"
        "- If the user says 'not X' or 'avoid X', do NOT set formality to X.\n"
        "  Example: 'not too formal' should NOT become 'formal' (prefer smart_casual or business).\n\n"
        "Optional intent flags (booleans):\n"
        "- intent_summer_lightweight\n"
        "- intent_rainy_or_cold\n"
        "- intent_polished\n"
        "- intent_not_sporty\n\n"
        "semantic_query (REQUIRED string):\n"
        "- One concise English line for embedding / dense retrieval (about 8–25 words).\n"
        "- Summarize shopping intent in clear, positive terms (e.g. 'non-athletic everyday' not just repeating 'not sporty').\n"
        "- Include men + garment/vibe keywords aligned with required_roles and intents. No markdown, no JSON inside the string.\n\n"
        "Few-shot examples (JSON only, abbreviated fields):\n"
        "Example 1\n"
        "User: \"Need something professional-looking, but keep it relaxed\"\n"
        "Return:\n"
        "{"
        "\"semantic_query\":\"men smart casual work outfit polished non-athletic relaxed\","
        "\"target_group\":\"men\","
        "\"required_roles\":[\"top\",\"bottom\",\"shoes\"],"
        "\"formality\":\"smart_casual\","
        "\"occasion\":\"work\","
        "\"intent_polished\":true,"
        "\"intent_not_sporty\":true"
        "}\n\n"
        "Example 2\n"
        "User: \"Outfit for warm weather with trainers\"\n"
        "Return:\n"
        "{"
        "\"semantic_query\":\"men lightweight summer outfit trainers sneakers breathable\","
        "\"target_group\":\"men\","
        "\"required_roles\":[\"top\",\"bottom\",\"shoes\"],"
        "\"preferred_categories\":[\"sneakers\"],"
        "\"formality\":null,"
        "\"occasion\":null,"
        "\"intent_summer_lightweight\":true"
        "}\n\n"
        "Example 3\n"
        "User: \"Wet weather outfit — include an outer layer, hood preferred\"\n"
        "Return:\n"
        "{"
        "\"semantic_query\":\"men rainy cold weather layered outfit jacket hood outerwear\","
        "\"target_group\":\"men\","
        "\"required_roles\":[\"top\",\"bottom\",\"shoes\",\"outerwear\"],"
        "\"requested_roles\":[\"outerwear\"],"
        "\"preferred_categories\":[\"jacket\"],"
        "\"formality\":null,"
        "\"occasion\":null,"
        "\"intent_rainy_or_cold\":true"
        "}\n\n"
        "Example 4\n"
        "User: \"Want a warm outer layer, avoid gym/athletic vibes\"\n"
        "Return:\n"
        "{"
        "\"semantic_query\":\"men warm casual jacket outfit everyday non-athletic\","
        "\"target_group\":\"men\","
        "\"required_roles\":[\"top\",\"bottom\",\"shoes\",\"outerwear\"],"
        "\"preferred_categories\":[\"jacket\"],"
        "\"formality\":\"casual\","
        "\"occasion\":\"casual\","
        "\"intent_not_sporty\":true"
        "}\n\n"
        "If the request is ambiguous, keep the deterministic fallback value."
    )
    payload = {
        "task": "refine_recommendation_query_constraints",
        "user_query": user_query,
        "deterministic_constraints": deterministic_constraints,
    }
    return _call_openai_json(
        model=settings.openai_model_query_parser,
        instructions=instructions,
        payload=payload,
        max_output_tokens=500,
    )


def llm_rerank_outfits(
    user_query: str,
    constraints: dict[str, Any],
    outfit_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Use OpenAI to rerank a small shortlist of outfit candidates."""

    instructions = (
        "You are a fashion outfit reranker. "
        "Return valid JSON only. Do not include markdown. "
        "Select and rank the best outfit candidates for the user query. "
        "Each candidate lists items with stable item_id fields. Treat item_id as the ground truth for identity.\n\n"
        "HARD CONSTRAINT (higher priority than stylistic preference):\n"
        "- Your first three entries in ranked_outfit_ids MUST name three outfits whose item_id sets are pairwise disjoint: "
        "no item_id may appear in more than one of those three outfits.\n"
        "- Before finalizing ranked_outfit_ids, mentally verify: build the union of item_id from outfit A, B, C — "
        "the size of that union must equal (len(A items)+len(B items)+len(C items)). If not, you picked overlapping outfits; replace one with a different candidate id from the shortlist.\n"
        "- If three fully disjoint outfits do not exist in the shortlist, still output three outfit_ids: pick the triple that minimizes item_id overlap; never keep an overlap when the shortlist contains any alternative outfit that removes it.\n\n"
        "Style ranking (after the constraint above):\n"
        "Prefer outfits that fit the query's requested tone, color direction, and category hints. "
        "Use the constraints fields as hard guidance. In particular:\n"
        "- If intent_summer_lightweight=true: prefer lighter-looking combinations; penalize sweaters/hoodies and heavy outerwear.\n"
        "- If intent_rainy_or_cold=true: penalize shorts/sandals; prefer outerwear and more practical footwear.\n"
        "- If intent_polished=true: penalize shorts; prefer shirts/trousers/boots and avoid section_theme=sport.\n"
        "- If intent_not_sporty=true: penalize section_theme=sport and overly athletic pieces.\n"
        "The top 3 should be meaningfully different: avoid near-duplicates that only swap one very similar item when disjoint alternatives exist.\n"
        "Keep explanations short, concrete, and comparative; you may note when you chose a disjoint triple. "
        "Use this output shape: "
        '{"ranked_outfit_ids":["outfit_1","outfit_2","outfit_3"],"explanations":{"outfit_1":"...","outfit_2":"...","outfit_3":"..."}}'
    )
    payload = {
        "task": "rerank_outfit_shortlist",
        "user_query": user_query,
        "constraints": constraints,
        "outfit_candidates": outfit_candidates,
    }
    return _call_openai_json(
        model=settings.openai_model_reranker,
        instructions=instructions,
        payload=payload,
        max_output_tokens=700,
    )


def llm_compose_outfits(
    user_query: str,
    constraints: dict[str, Any],
    candidate_pools_by_role: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Ask the LLM to pick item_ids from per-role pools to build exactly 3 outfits (grounded)."""

    instructions = (
        "You are a fashion stylist for a recommender with a fixed catalog. "
        "Return valid JSON only. Do not include markdown.\n\n"
        "You MUST build exactly 3 outfit objects in the `outfits` array.\n"
        "Each outfit MUST include every key listed in constraints.required_roles exactly once, "
        "mapping role name -> item_id string chosen from that role's candidate list in candidate_pools_by_role.\n"
        "Do NOT invent item_ids. Every item_id you output must appear verbatim in the corresponding role pool.\n\n"
        "HARD CONSTRAINT:\n"
        "- Across all 3 outfits, no item_id may repeat. "
        "If the pools cannot support 3 fully disjoint outfits with the required roles, "
        "still output 3 outfits but minimize overlap; prefer swapping to a different id from the same role pool.\n\n"
        "PRIORITY: visual harmony and style (choose item_ids to make each outfit feel intentional, not random):\n"
        "- Treat normalized_color as your main tool for cohesion. Prefer combinations where pieces feel like they belong together.\n"
        "- Strong approaches when the catalog allows: (1) tonal / monochrome — same color family with lighter/darker shades; "
        "(2) neutral base — black, white, grey, navy, beige, cream anchoring most pieces with one subtle accent; "
        "(3) analogous warmth — browns/tans/olive, or cool blues/greys together; "
        "(4) one clear accent color against neutrals (do not mix many loud colors in one outfit).\n"
        "- Avoid looking 'thrown together': clashy unrelated hues, or every piece a different loud color, unless the user clearly asks for maximal color.\n"
        "- If formality or intent_polished is in play, favor restrained palettes and avoid jarring sport + dressy color clashes unless clearly supported.\n"
        "- Use display_name and section_theme as tie-breakers after color: keep athletic vs tailored vibes consistent within one outfit.\n"
        "- Respect constraints and preferred_colors: if the user asked for a color direction, lean the whole outfit that way.\n"
        "- The 3 outfits should differ in mood or palette when the pools allow (e.g. one neutral look, one warmer, one cooler), "
        "but each single outfit must stay internally harmonious.\n"
        "- In each explanation, briefly note the palette logic (e.g. 'navy + grey tonal', 'beige chinos anchor olive and cream').\n\n"
        "Output shape exactly:\n"
        '{"outfits":[{"items_by_role":{"<role>":"<item_id>", "...":"..."},"explanation":"one short sentence"},'
        '{"items_by_role":{...},"explanation":"..."},{"items_by_role":{...},"explanation":"..."}]}'
    )
    payload = {
        "task": "compose_outfits_from_pools",
        "user_query": user_query,
        "constraints": constraints,
        "candidate_pools_by_role": candidate_pools_by_role,
    }
    return _call_openai_json(
        model=settings.openai_model_reranker,
        instructions=instructions,
        payload=payload,
        max_output_tokens=900,
    )


def llm_judge_retrieval(
    *,
    user_query: str,
    sparse_only: dict[str, Any],
    sparse_plus_dense: dict[str, Any],
) -> dict[str, Any] | None:
    """Use an LLM to judge which retrieval result better matches the query."""

    instructions = (
        "You are a strict retrieval evaluator for a fashion recommender. "
        "You will compare two retrieval outputs for the same user query. "
        "Return valid JSON only. Do not include markdown. "
        "Do NOT judge the writing quality of explanations. Ignore explanation prose completely.\n"
        "Judge ONLY the structured evidence: item display_name, category, role, color, section_theme, and scores.\n\n"
        "Scoring rubric (0-5 each):\n"
        "- relevance: matches the user intent (warm/cozy/polished/lightweight/etc.)\n"
        "- constraint_fit: respects explicit constraints (category mentions, 'not X', office vs sporty, etc.)\n"
        "- coherence: the outfit combinations make sense together\n"
        "overall = relevance + constraint_fit + coherence (0-15)\n\n"
        "Reasoning rules:\n"
        "- If two outputs are close, pick tie.\n"
        "- In reasons, cite concrete evidence (e.g., 'includes shorts for rainy day', 'section_theme sport', etc.).\n\n"
        "Output JSON format:\n"
        "{"
        "\"winner\":\"sparse_only|sparse_plus_dense|tie\","
        "\"scores\":{"
        "\"sparse_only\":{\"relevance\":0,\"constraint_fit\":0,\"coherence\":0,\"overall\":0},"
        "\"sparse_plus_dense\":{\"relevance\":0,\"constraint_fit\":0,\"coherence\":0,\"overall\":0}"
        "},"
        "\"reasons\":[\"...\"]"
        "}"
    )

    payload = {
        "task": "judge_retrieval_outputs",
        "user_query": user_query,
        "candidate_a": {"name": "sparse_only", **sparse_only},
        "candidate_b": {"name": "sparse_plus_dense", **sparse_plus_dense},
    }
    return _call_openai_json(
        model=settings.openai_model_judge,
        instructions=instructions,
        payload=payload,
        max_output_tokens=500,
    )


def llm_score_retrieval(
    *,
    user_query: str,
    retrieval_output: dict[str, Any],
) -> dict[str, Any] | None:
    """Score a single retrieval output (no pairwise comparison)."""

    instructions = (
        "You are a strict retrieval evaluator for a fashion recommender. "
        "You will score ONE retrieval output for the given user query. "
        "Return valid JSON only. Do not include markdown.\n\n"
        "Ignore explanation prose completely. "
        "Judge ONLY structured evidence: item display_name, category, role, color, section_theme, and scores.\n\n"
        "If llm_status indicates an LLM reranker was used, you may still judge the output, but focus on the outfit items.\n\n"
        "Scoring rubric (0-5 each):\n"
        "- relevance: matches the user intent (warm/cozy/polished/lightweight/etc.)\n"
        "- constraint_fit: respects explicit constraints (category mentions, 'not X', office vs sporty, etc.)\n"
        "- coherence: the outfit combinations make sense together\n"
        "overall = relevance + constraint_fit + coherence (0-15)\n\n"
        "Output JSON format:\n"
        "{"
        "\"scores\":{\"relevance\":0,\"constraint_fit\":0,\"coherence\":0,\"overall\":0},"
        "\"reasons\":[\"...\"]"
        "}"
    )

    payload = {
        "task": "score_retrieval_output",
        "user_query": user_query,
        "retrieval_output": retrieval_output,
    }
    return _call_openai_json(
        model=settings.openai_model_judge,
        instructions=instructions,
        payload=payload,
        max_output_tokens=450,
    )
