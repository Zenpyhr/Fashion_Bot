"""Postgres + pgvector helpers for storing and fetching embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import insert

from src.shared.config import settings


metadata = MetaData()


def create_engine_from_settings():
    """Create a SQLAlchemy engine; fail reasonably fast if Postgres is not running."""

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )


def ensure_pgvector_extension(engine) -> None:
    """Create the pgvector extension if missing."""

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def infer_vector_dim(embedding_model: str) -> int:
    if embedding_model == "text-embedding-3-small":
        return 1536
    if embedding_model == "text-embedding-3-large":
        return 3072
    raise RuntimeError(f"Unknown embedding dimension for model={embedding_model!r}.")


def normalize_item_text(text_value: str) -> str:
    """Light normalization used consistently for embedding + hashing."""

    return " ".join(str(text_value or "").split()).strip().lower()


def sha256_text(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def build_item_text(row: dict) -> str:
    """Build the canonical text used to embed an item."""

    parts = [
        row.get("display_name", ""),
        row.get("description", ""),
        row.get("normalized_category", ""),
        row.get("normalized_pattern", ""),
        row.get("section_theme", ""),
    ]
    normalized_parts = [normalize_item_text(part) for part in parts if part]
    return "\n".join([part for part in normalized_parts if part])


def get_catalog_item_embeddings_table(vector_dim: int) -> Table:
    """Define the catalog embeddings table with the selected vector dimension."""

    existing = metadata.tables.get(settings.catalog_item_embeddings_table)
    if existing is not None:
        return existing

    return Table(
        settings.catalog_item_embeddings_table,
        metadata,
        Column("item_id", String, primary_key=True),
        Column("embedding", Vector(vector_dim), nullable=False),
        Column("embedding_model", String, nullable=False),
        Column("text_hash", String(64), nullable=False),
        Column("item_text", Text, nullable=False),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )


def get_wardrobe_item_embeddings_table(vector_dim: int) -> Table:
    """Define the wardrobe embeddings table with the selected vector dimension."""

    existing = metadata.tables.get(settings.wardrobe_item_embeddings_table)
    if existing is not None:
        return existing

    return Table(
        settings.wardrobe_item_embeddings_table,
        metadata,
        Column("user_id", String, primary_key=True),
        Column("wardrobe_item_id", String, primary_key=True),
        Column("embedding", Vector(vector_dim), nullable=False),
        Column("embedding_model", String, nullable=False),
        Column("text_hash", String(64), nullable=False),
        Column("item_text", Text, nullable=False),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )


@dataclass(frozen=True)
class EmbeddingRow:
    item_id: str
    embedding: list[float]
    embedding_model: str
    text_hash: str
    item_text: str


@dataclass(frozen=True)
class WardrobeEmbeddingRow:
    user_id: str
    wardrobe_item_id: str
    embedding: list[float]
    embedding_model: str
    text_hash: str
    item_text: str


def ensure_embeddings_table(engine, vector_dim: int) -> Table:
    """Ensure the catalog embeddings table exists."""

    ensure_pgvector_extension(engine)
    table = get_catalog_item_embeddings_table(vector_dim)
    metadata.create_all(engine, tables=[table])
    return table


def ensure_wardrobe_embeddings_table(engine, vector_dim: int) -> Table:
    """Ensure the wardrobe embeddings table exists."""

    ensure_pgvector_extension(engine)
    table = get_wardrobe_item_embeddings_table(vector_dim)
    metadata.create_all(engine, tables=[table])
    return table


def fetch_existing_text_hashes(engine, table: Table, item_ids: Iterable[str]) -> dict[str, str]:
    ids = list({str(x) for x in item_ids})
    if not ids:
        return {}
    with engine.begin() as conn:
        rows = conn.execute(
            table.select().with_only_columns(table.c.item_id, table.c.text_hash).where(table.c.item_id.in_(ids))
        ).fetchall()
    return {str(item_id): str(text_hash) for item_id, text_hash in rows}


def fetch_embeddings(engine, table: Table, item_ids: Iterable[str]) -> dict[str, list[float]]:
    ids = list({str(x) for x in item_ids})
    if not ids:
        return {}
    with engine.begin() as conn:
        rows = conn.execute(
            table.select().with_only_columns(table.c.item_id, table.c.embedding).where(table.c.item_id.in_(ids))
        ).fetchall()

    # pgvector returns a list/array-like; coerce to list[float]
    return {str(item_id): list(embedding) for item_id, embedding in rows}


def fetch_wardrobe_embeddings(
    engine,
    table: Table,
    *,
    user_id: str,
    wardrobe_item_ids: Iterable[str],
) -> dict[str, list[float]]:
    ids = list({str(x) for x in wardrobe_item_ids})
    if not ids:
        return {}

    with engine.begin() as conn:
        rows = conn.execute(
            table.select()
            .with_only_columns(table.c.wardrobe_item_id, table.c.embedding)
            .where(table.c.user_id == user_id)
            .where(table.c.wardrobe_item_id.in_(ids))
        ).fetchall()

    return {str(item_id): list(embedding) for item_id, embedding in rows}

def upsert_wardrobe_embeddings(engine, table: Table, rows: Iterable[WardrobeEmbeddingRow]) -> int:
    payload = [
        {
            "user_id": row.user_id,
            "wardrobe_item_id": row.wardrobe_item_id,
            "embedding": row.embedding,
            "embedding_model": row.embedding_model,
            "text_hash": row.text_hash,
            "item_text": row.item_text,
        }
        for row in rows
    ]

    if not payload:
        return 0

    stmt = insert(table).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=[table.c.user_id, table.c.wardrobe_item_id],
        set_={
            "embedding": stmt.excluded.embedding,
            "embedding_model": stmt.excluded.embedding_model,
            "text_hash": stmt.excluded.text_hash,
            "item_text": stmt.excluded.item_text,
            "updated_at": func.now(),
        },
    )

    with engine.begin() as conn:
        conn.execute(stmt)

    return len(payload)


