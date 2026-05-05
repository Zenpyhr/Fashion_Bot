"""Embedding helpers."""

from __future__ import annotations

from typing import Iterable

from src.integrations.openai_client import create_openai_client, openai_is_configured
from src.shared.config import settings


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI embeddings API."""

    if not openai_is_configured():
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY to embed texts.")

    if not texts:
        return []

    client = create_openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_text_openai(text: str) -> list[float]:
    return embed_texts_openai([text])[0]


def l2_normalize(vec: Iterable[float]) -> list[float]:
    values = [float(x) for x in vec]
    norm = sum(v * v for v in values) ** 0.5
    if norm <= 0:
        return values
    return [v / norm for v in values]
