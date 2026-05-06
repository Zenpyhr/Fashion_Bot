"""Convenience wrapper: build QA Chroma index.

Preferred entrypoint for Track A after the repo refactor to `data/qa/` and `src/qa/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa.scripts.build_db import main as build_main


if __name__ == "__main__":
    build_main()

