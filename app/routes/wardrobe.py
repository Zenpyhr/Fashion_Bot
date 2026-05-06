"""Wardrobe upload + management API routes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from src.database.wardrobe_store import create_engine_from_settings, ensure_wardrobe_items_table
from src.integrations.pgvector_store import ensure_wardrobe_embeddings_table, infer_vector_dim
from src.shared.config import settings
from src.recommender.wardrobe_service import ingest_wardrobe_image

router = APIRouter()


@router.post("/upload")
async def upload_wardrobe_item(
    user_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict:
    """Upload a single image and ingest it as a wardrobe item."""

    suffix = Path(image.filename or "").suffix
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await image.read()
            tmp.write(content)

        result = ingest_wardrobe_image(user_id=user_id, source_image_path=tmp_path)
        return {
            "ok": True,
            "user_id": user_id,
            "wardrobe_item_id": result.wardrobe_item_id,
            "staged_image_path": result.staged_image_path,
            "quarantine_reasons": result.quarantine_reasons,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.post("/clear")
async def clear_wardrobe(user_id: str = Form(...)) -> dict:
    """Delete all wardrobe items for a user (metadata + embeddings)."""

    engine = create_engine_from_settings()

    items_table = ensure_wardrobe_items_table(engine)
    emb_table = ensure_wardrobe_embeddings_table(engine, vector_dim=infer_vector_dim(settings.openai_embedding_model))

    with engine.begin() as conn:
        deleted_items = conn.execute(items_table.delete().where(items_table.c.user_id == user_id)).rowcount
        deleted_embeddings = conn.execute(emb_table.delete().where(emb_table.c.user_id == user_id)).rowcount

    return {
        "ok": True,
        "user_id": user_id,
        "deleted_items": int(deleted_items or 0),
        "deleted_embeddings": int(deleted_embeddings or 0),
    }


