from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import benchmarks as benchmark_routes


class _FakeRepository:
    def export_json(self):
        return [{"id": 1, "request_id": "req_test"}]

    def export_csv(self):
        return "id,request_id\r\n1,req_test\r\n"


def _build_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(benchmark_routes, "get_repository", lambda: _FakeRepository())
    app = FastAPI()
    app.include_router(benchmark_routes.router)
    return TestClient(app)


def test_export_json_route_is_not_shadowed(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.get("/api/benchmarks/export?format=json")

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "request_id": "req_test"}]


def test_export_csv_route_is_not_shadowed(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.get("/api/benchmarks/export?format=csv")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "id,request_id" in response.text
