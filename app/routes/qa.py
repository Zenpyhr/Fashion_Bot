"""QA API routes."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.qa.scripts import query_answer
from src.shared.schemas import QARequest, QAResponse

router = APIRouter()


def _is_insufficient_evidence_answer(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    markers = [
        "i do not have enough reliable evidence in the retrieved sources",
        "no supported trend can be concluded from the retrieved sources",
        "does not contain enough direct, relevant evidence for the user question",
        "retrieved sources are insufficient to support a direct answer to this question",
    ]
    return any(marker in normalized for marker in markers)


def _extract_citations(answer: str, max_source_id: int) -> list[str]:
    citation_pattern = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
    seen: set[int] = set()
    ordered_ids: list[int] = []

    for match in citation_pattern.finditer(answer):
        source_id = int(match.group(1))
        if source_id < 1 or source_id > max_source_id:
            continue
        if source_id in seen:
            continue
        seen.add(source_id)
        ordered_ids.append(source_id)

    return [f"Source {source_id}" for source_id in ordered_ids]


def _build_sources(contexts: list[dict]) -> list[dict]:
    sources = []
    for idx, item in enumerate(contexts, start=1):
        sources.append(
            {
                "source_id": f"Source {idx}",
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "scope": item.get("scope", ""),
                "article_id": item.get("article_id", ""),
                "excerpt": str(item.get("text", ""))[:320],
            }
        )
    return sources


@router.post("", response_model=QAResponse)
def ask_fashion_question(payload: QARequest) -> QAResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    db_path = Path(query_answer.db).resolve()
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "QA index is not built yet. "
                "Run `python scripts/qa_build_db.py` from the project root first."
            ),
        )

    try:
        contexts, scope_decision = query_answer.retrieve(
            question=question,
            top_k=query_answer.default_top_k,
            db_path=str(db_path),
            return_scope_decision=True,
            retrieval_strategy="mix",
        )
        prompt = query_answer.llm_prompt(
            question=question,
            contexts=contexts,
            detected_scopes=scope_decision.get("top_scopes", []),
        )
        answer = query_answer.generate_answer(prompt, source_count=len(contexts))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"QA generation failed: {exc}") from exc

    if _is_insufficient_evidence_answer(answer):
        return QAResponse(answer=answer, citations=[], sources=[])

    citations = _extract_citations(answer, max_source_id=len(contexts))
    sources = _build_sources(contexts)
    return QAResponse(answer=answer, citations=citations, sources=sources)
