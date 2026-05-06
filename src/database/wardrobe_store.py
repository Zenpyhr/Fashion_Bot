"""Postgres storage for user wardrobe items (Option 2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
)

from src.shared.config import settings


metadata = MetaData()


def create_engine_from_settings():
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )


def ensure_wardrobe_items_table(engine) -> Table:
    """Create the wardrobe_items table if missing."""

    existing = metadata.tables.get("wardrobe_items")
    if existing is not None:
        return existing

    table = Table(
        "wardrobe_items",
        metadata,
        Column("user_id", String, primary_key=True),
        Column("wardrobe_item_id", String, primary_key=True),
        Column("source_item_id", String, nullable=True),
        Column("source_type", String, nullable=False, server_default=text("'wardrobe'")),
        Column("status", String, nullable=False, server_default=text("'tagged'")),
        Column("error_message", Text, nullable=True),
        Column("content_hash", String(64), nullable=True),
        Column("image_path", Text, nullable=True),
        Column("image_relative_path", Text, nullable=True),
        Column("display_name", Text, nullable=True),
        Column("description", Text, nullable=True),
        Column("target_group", String, nullable=True),
        Column("recommendation_role", String, nullable=True),
        Column("normalized_category", String, nullable=True),
        Column("normalized_color", String, nullable=True),
        Column("normalized_pattern", String, nullable=True),
        Column("section_theme", String, nullable=True),
        Column("raw_vlm_json", JSON, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    )

    metadata.create_all(engine, tables=[table])
    return table


@dataclass(frozen=True)
class WardrobeRow:
    user_id: str
    wardrobe_item_id: str
    source_item_id: str | None
    content_hash: str | None
    item: dict[str, Any]


def upsert_wardrobe_items(engine, table: Table, rows: Iterable[WardrobeRow]) -> int:
    """Upsert wardrobe items. Returns number of attempted rows."""

    payload = []
    for row in rows:
        item = row.item
        payload.append(
            {
                "user_id": row.user_id,
                "wardrobe_item_id": row.wardrobe_item_id,
                "source_item_id": row.source_item_id,
                "source_type": str(item.get("source_type") or "wardrobe"),
                "status": str(item.get("status") or "tagged"),
                "error_message": item.get("error_message"),
                "content_hash": row.content_hash,
                "image_path": item.get("image_path"),
                "image_relative_path": item.get("image_relative_path"),
                "display_name": item.get("display_name"),
                "description": item.get("description"),
                "target_group": item.get("target_group"),
                "recommendation_role": item.get("recommendation_role"),
                "normalized_category": item.get("normalized_category"),
                "normalized_color": item.get("normalized_color"),
                "normalized_pattern": item.get("normalized_pattern"),
                "section_theme": item.get("section_theme"),
                "raw_vlm_json": item.get("raw_vlm_json"),
            }
        )

    if not payload:
        return 0

    # Postgres upsert.
    stmt = insert(table).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=[table.c.user_id, table.c.wardrobe_item_id],
        set_={
            "source_item_id": stmt.excluded.source_item_id,
            "source_type": stmt.excluded.source_type,
            "status": stmt.excluded.status,
            "error_message": stmt.excluded.error_message,
            "content_hash": stmt.excluded.content_hash,
            "image_path": stmt.excluded.image_path,
            "image_relative_path": stmt.excluded.image_relative_path,
            "display_name": stmt.excluded.display_name,
            "description": stmt.excluded.description,
            "target_group": stmt.excluded.target_group,
            "recommendation_role": stmt.excluded.recommendation_role,
            "normalized_category": stmt.excluded.normalized_category,
            "normalized_color": stmt.excluded.normalized_color,
            "normalized_pattern": stmt.excluded.normalized_pattern,
            "section_theme": stmt.excluded.section_theme,
            "raw_vlm_json": stmt.excluded.raw_vlm_json,
            "updated_at": func.now(),
        },
    )

    with engine.begin() as conn:
        conn.execute(stmt)
    return len(payload)


def fetch_wardrobe_items_for_user(engine, table: Table, *, user_id: str) -> list[dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            select(table).where(table.c.user_id == user_id).where(table.c.status == "tagged")
        ).mappings().fetchall()

    # Return as item dicts, mirroring catalog row-ish shape.
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "item_id": row["wardrobe_item_id"],
                "source_type": row.get("source_type") or "wardrobe",
                "image_path": row.get("image_path"),
                "image_relative_path": row.get("image_relative_path"),
                "display_name": row.get("display_name"),
                "description": row.get("description"),
                "target_group": row.get("target_group"),
                "recommendation_role": row.get("recommendation_role"),
                "normalized_category": row.get("normalized_category"),
                "normalized_color": row.get("normalized_color"),
                "normalized_pattern": row.get("normalized_pattern"),
                "section_theme": row.get("section_theme"),
            }
        )
    return items

