import importlib
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


def _load_app_with_stubbed_qa(monkeypatch) -> object:
    """Insert a small fake QA module, then import app.main using that stub.

    This lets the API tests construct the FastAPI app without importing the real
    QA retrieval stack.
    """

    fake_query_answer = types.ModuleType("query_answer")
    fake_query_answer.db = str(Path("data/qa/index/fashion_chroma_db"))
    fake_query_answer.default_top_k = 5
    fake_query_answer.retrieve = lambda *args, **kwargs: ([], {"top_scopes": []})
    fake_query_answer.llm_prompt = lambda *args, **kwargs: ""
    fake_query_answer.generate_answer = (
        lambda *args, **kwargs: "I do not have enough reliable evidence in the retrieved sources."
    )

    monkeypatch.setitem(sys.modules, "src.qa.scripts.query_answer", fake_query_answer)
    sys.modules.pop("app.routes.qa", None)
    sys.modules.pop("app.main", None)

    module = importlib.import_module("app.main")
    return module.app


def test_health_and_core_routes_exist(monkeypatch) -> None:
    app = _load_app_with_stubbed_qa(monkeypatch)

    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/qa" in paths
    assert "/recommend" in paths


def test_recommend_route_returns_structured_payload(monkeypatch) -> None:
    app = _load_app_with_stubbed_qa(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={"user_query": "Need a smart casual men's outfit in black."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parsed_constraints"]["target_group"] == "men"
    assert payload["parsed_constraints"]["formality"] == "smart_casual"
    assert "outfits" in payload
    assert "missing_items" in payload
    assert isinstance(payload["outfits"], list)
    assert isinstance(payload["explanations"], list)

    if payload["outfits"]:
        first_outfit = payload["outfits"][0]
        assert "score" in first_outfit
        assert "items" in first_outfit
        assert "explanation" in first_outfit
