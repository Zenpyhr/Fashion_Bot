"""Quick DB check for catalog embeddings table.

Usage:
  python scripts/check_catalog_embeddings_db.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.pgvector_store import create_engine_from_settings
from src.shared.config import settings


def main() -> None:
    engine = create_engine_from_settings()
    table = settings.catalog_item_embeddings_table
    with engine.begin() as conn:
        ext = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).fetchall()
        exists = conn.execute(
            text(
                "SELECT to_regclass(:table_name)"
            ),
            {"table_name": table},
        ).scalar_one()
        count = None
        if exists:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

    print("pgvector_extension_installed:", bool(ext))
    print("embeddings_table:", table)
    print("table_exists:", bool(exists))
    print("row_count:", count)


if __name__ == "__main__":
    main()

