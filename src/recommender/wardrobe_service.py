"""Wardrobe ingestion service.

Single entrypoint that both scripts and future UI upload routes should call.

Responsibilities:
- stage image into data/recommender/user_wardrobe/<user_id>/uploads/<wardrobe_item_id>.<ext>
- tag with VLM
- normalize into canonical recommender vocab
- upsert into Postgres wardrobe_items table
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.database.wardrobe_store import (
    WardrobeRow,
    create_engine_from_settings,
    ensure_wardrobe_items_table,
    upsert_wardrobe_items,
)
from src.recommender.vlm_tagging import tag_image
from src.recommender.wardrobe_normalize import normalize_wardrobe_item


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARDROBE_ROOT = PROJECT_ROOT / "data" / "recommender" / "user_wardrobe"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class IngestResult:
    wardrobe_item_id: str
    staged_image_path: str
    quarantine_reasons: list[str]


def ingest_wardrobe_image(
    *,
    user_id: str,
    source_image_path: str,
    seed: dict[str, Any] | None = None,
    model: str = "gpt-5.4-mini",
    detail: str = "auto",
) -> IngestResult:
    """Ingest one wardrobe image into Postgres.

    Note: `wardrobe_item_id` is generated from the image content hash, regardless
    of the original filename (e.g. H&M ids in Images/).
    """

    src = Path(source_image_path)
    if not src.exists():
        raise FileNotFoundError(f"Wardrobe image not found: {src}")

    content_hash = _sha256_file(src)
    wardrobe_item_id = content_hash[:16]  # short stable id for UI/URLs
    ext = (src.suffix or ".jpg").lower()

    dest_dir = WARDROBE_ROOT / user_id / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{wardrobe_item_id}{ext}"

    if not dest.exists():
        shutil.copyfile(src, dest)

    # VLM tagging uses the staged path.
    vlm = tag_image(str(dest), seed=seed, model=model, detail=detail)
    # Preserve raw output for debugging / future remapping.
    vlm_with_raw = dict(vlm)
    vlm_with_raw["raw_vlm_json"] = json.loads(json.dumps(vlm, ensure_ascii=False))

    normalized = normalize_wardrobe_item(vlm_with_raw)

    # Upsert into DB.
    engine = create_engine_from_settings()
    table = ensure_wardrobe_items_table(engine)

    row = WardrobeRow(
        user_id=user_id,
        wardrobe_item_id=wardrobe_item_id,
        source_item_id=src.stem,
        content_hash=content_hash,
        item={
            **normalized.item,
            "status": "tagged" if not normalized.quarantine_reasons else "tagged",
            "error_message": None,
        },
    )
    upsert_wardrobe_items(engine, table, [row])

    return IngestResult(
        wardrobe_item_id=wardrobe_item_id,
        staged_image_path=str(dest),
        quarantine_reasons=normalized.quarantine_reasons,
    )

