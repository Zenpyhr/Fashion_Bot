"""Vertex AI integration helpers."""

from src.shared.config import settings


def vertex_runtime_config() -> dict[str, str]:
    return {
        "project": settings.google_cloud_project,
        "location": settings.google_cloud_location,
        "text_model": settings.vertex_model_text,
        "vision_model": settings.vertex_model_vision,
        "embedding_model": settings.embedding_model,
    }
