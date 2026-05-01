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

    try:
        client = create_openai_client()
        response = client.responses.create(
            model=model,
            reasoning={"effort": settings.openai_reasoning_effort},
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
        "You are a fashion recommendation query parser. "
        "Return valid JSON only. Do not include markdown. "
        "Keep the schema shape aligned with the provided deterministic constraints. "
        "Use only these roles: top, bottom, shoes, outerwear. "
        "Use only these target groups: women, men. "
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
