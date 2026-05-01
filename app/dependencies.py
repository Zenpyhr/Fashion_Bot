"""Shared FastAPI dependencies."""

from src.shared.config import settings


def get_settings():
    return settings
