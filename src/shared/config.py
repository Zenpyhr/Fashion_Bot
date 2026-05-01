"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_cloud_project: str = "your-gcp-project-id"
    google_cloud_location: str = "us-central1"
    vertex_model_text: str = "gemini-2.5-flash"
    vertex_model_vision: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot"
    items_table: str = "items"
    article_chunks_table: str = "article_chunks"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
