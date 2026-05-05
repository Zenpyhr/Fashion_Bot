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
        max_output_tokens=400,
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
        "Prefer outfits that fit the query's requested tone, color direction, and category hints. "
        "The top 3 should be meaningfully different when possible; avoid near-duplicate outfits that only swap one very similar item. "
        "Penalize repetitive selections unless the shortlist truly has no better variation. "
        "Keep explanations short, concrete, and comparative. "
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
