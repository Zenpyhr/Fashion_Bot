"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model_query_parser: str = "gpt-5.4-mini"
    openai_model_reranker: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_reasoning_effort: str = "low"
    enable_openai_query_parser: bool = True
    enable_openai_reranker: bool = True
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot"
    items_table: str = "items"
    article_chunks_table: str = "article_chunks"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
