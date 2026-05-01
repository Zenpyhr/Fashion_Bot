from app.main import app


def test_health_route_exists() -> None:
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/qa" in paths
    assert "/recommend" in paths
