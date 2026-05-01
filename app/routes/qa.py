"""QA API routes."""

from fastapi import APIRouter

from src.shared.schemas import QARequest, QAResponse

router = APIRouter()


@router.post("", response_model=QAResponse)
def ask_fashion_question(payload: QARequest) -> QAResponse:
    return QAResponse(
        answer=f"QA pipeline not implemented yet for: {payload.question}",
        citations=[],
        sources=[],
    )
