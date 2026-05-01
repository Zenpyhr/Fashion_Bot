from src.shared.schemas import QARequest


def test_qa_request_schema() -> None:
    payload = QARequest(question="What is business casual?")
    assert payload.question
