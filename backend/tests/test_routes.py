"""Test FastAPI routes."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "uptime_seconds" in data


def test_session_start_stop():
    r = client.post("/session/start", json={"media_type": "omni", "language": "zh"})
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    r = client.post("/session/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"
