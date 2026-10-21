from __future__ import annotations

import jwt
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_token_is_a_room_scoped_jwt():
    r = client.post("/api/token", json={"name": "Sample Caller"})
    assert r.status_code == 200
    body = r.json()
    claims = jwt.decode(body["token"], options={"verify_signature": False})
    assert claims["video"]["room"] == body["room"] and claims["video"]["roomJoin"] is True
    assert claims["sub"] == body["identity"]


def test_token_rejects_bad_input():
    assert client.post("/api/token", json={"name": ""}).status_code == 422


def test_metrics_count_tokens_and_requests():
    client.post("/api/token", json={"name": "A"})
    text = client.get("/metrics").text
    assert "voicedesk_tokens_issued_total" in text and "voicedesk_http_request_seconds" in text


def test_index_and_config():
    assert "VoiceDesk" in client.get("/").text
    assert client.get("/api/config").json()["version"]
