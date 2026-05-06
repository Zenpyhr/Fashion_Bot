"""Convenience wrapper: run QA answer from CLI.

Example:
  python scripts\\qa_answer.py \"What are the top Spring 2026 trends?\"
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa.scripts.query_answer import qa_answer


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Provide a question string.")
    question = " ".join(sys.argv[1:]).strip()
    print(qa_answer(question))


if __name__ == "__main__":
    main()

