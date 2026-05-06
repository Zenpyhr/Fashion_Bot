"""Ingest 10 demo wardrobe images into Postgres.

This stages images into `data/recommender/user_wardrobe/<user_id>/uploads/` with new IDs
(content-hash based), tags them with VLM, normalizes, and upserts into the
`wardrobe_items` table.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.wardrobe_service import ingest_wardrobe_image


DEMO_USER_ID = "demo_user"

DEMO_IMAGES = [
    r"Images\010\0108775015.jpg",
    r"Images\010\0108775044.jpg",
    r"Images\010\0108775051.jpg",
    r"Images\011\0110065001.jpg",
    r"Images\011\0110065002.jpg",
    r"Images\011\0110065011.jpg",
    r"Images\011\0111565001.jpg",
    r"Images\011\0111565003.jpg",
    r"Images\011\0111586001.jpg",
    r"Images\011\0111593001.jpg",
]


def main() -> None:
    for path in DEMO_IMAGES:
        result = ingest_wardrobe_image(user_id=DEMO_USER_ID, source_image_path=path)
        print(f"- {path} -> wardrobe_item_id={result.wardrobe_item_id} staged={result.staged_image_path}")
        if result.quarantine_reasons:
            print(f"  quarantine: {', '.join(result.quarantine_reasons)}")


if __name__ == "__main__":
    main()

