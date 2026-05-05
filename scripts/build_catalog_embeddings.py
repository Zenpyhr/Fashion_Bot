"""Build / refresh dense embeddings for catalog items in Postgres (pgvector).

Usage (PowerShell):
  python scripts/build_catalog_embeddings.py

Notes:
- Requires DATABASE_URL to point to a Postgres instance.
- Requires OPENAI_API_KEY for embedding generation.
- Re-embeds only items whose normalized item_text changed (text_hash mismatch).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.embeddings import embed_texts_openai, l2_normalize
from src.integrations.pgvector_store import (
    build_item_text,
    create_engine_from_settings,
    ensure_embeddings_table,
    fetch_existing_text_hashes,
    sha256_text,
)
from src.shared.config import settings


def _infer_vector_dim() -> int:
    # Minimal mapping for the common OpenAI embedding models.
    # If you change models, ensure this mapping matches the provider's output dimension.
    model = settings.openai_embedding_model
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    raise RuntimeError(
        f"Unknown embedding dimension for model={model!r}. "
        "Add it to _infer_vector_dim() or set a supported model."
    )


def main() -> None:
    catalog_path = Path(settings.catalog_items_csv)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {catalog_path}")

    df = pd.read_csv(catalog_path)
    if "item_id" not in df.columns:
        raise ValueError("Catalog CSV must include an item_id column.")

    vector_dim = _infer_vector_dim()
    engine = create_engine_from_settings()
    table = ensure_embeddings_table(engine, vector_dim=vector_dim)

    rows = df.to_dict(orient="records")
    item_ids = [str(row["item_id"]) for row in rows]
    existing = fetch_existing_text_hashes(engine, table, item_ids)

    pending = []
    for row in rows:
        item_id = str(row["item_id"])
        item_text = build_item_text(row)
        text_hash = sha256_text(item_text)
        if existing.get(item_id) != text_hash:
            pending.append((item_id, item_text, text_hash))

    print(f"Catalog rows: {len(rows)}")
    print(f"Embeddings up-to-date: {len(rows) - len(pending)}")
    print(f"Needs embedding: {len(pending)}")

    if not pending:
        from sqlalchemy import select, func
        with engine.begin() as conn:
            count = conn.execute(select(func.count()).select_from(table)).scalar_one()
        print(f"DB table {table.name}: {count} rows")
        return

    batch_size = 128
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        item_ids_batch = [item_id for item_id, _, _ in batch]
        texts_batch = [item_text for _, item_text, _ in batch]
        hashes_batch = [text_hash for _, _, text_hash in batch]

        embeddings = embed_texts_openai(texts_batch)
        embeddings = [l2_normalize(vec) for vec in embeddings]

        payload = [
            {
                "item_id": item_id,
                "embedding": embedding,
                "embedding_model": settings.openai_embedding_model,
                "text_hash": text_hash,
                "item_text": item_text,
            }
            for item_id, item_text, text_hash, embedding in zip(
                item_ids_batch, texts_batch, hashes_batch, embeddings, strict=True
            )
        ]

        with engine.begin() as conn:
            for row_payload in payload:
                # Upsert by primary key (item_id)
                stmt = pg_insert(table).values(**row_payload)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[table.c.item_id],
                    set_=row_payload,
                )
                conn.execute(stmt)

        print(f"Upserted {len(payload)} embeddings ({start + len(batch)}/{len(pending)}).")

    from sqlalchemy import select, func
    with engine.begin() as conn:
        count = conn.execute(select(func.count()).select_from(table)).scalar_one()
    print(f"DB table {table.name}: {count} rows")


if __name__ == "__main__":
    main()

