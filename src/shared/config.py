"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model_query_parser: str = "gpt-4o-mini"
    openai_model_reranker: str = "gpt-4o-mini"
    openai_model_judge: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_reasoning_effort: str = "low"
    enable_openai_query_parser: bool = True
    enable_openai_reranker: bool = True
    # When True, LLM picks item_ids from per-role pools to form 3 outfits (grounded; no invented SKUs).
    enable_openai_combo_composer: bool = True
    combo_composer_max_items_per_role_for_llm: int = 18
    enable_dense_retrieval_rerank: bool = False
    dense_shortlist_k_per_role: int = 200
    dense_rerank_n_per_role: int = 50
    catalog_items_csv: str = "data/recommender/processed/catalog_items/catalog_items_demo.csv"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot"
    items_table: str = "items"
    article_chunks_table: str = "article_chunks"
    catalog_item_embeddings_table: str = "catalog_item_embeddings"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
