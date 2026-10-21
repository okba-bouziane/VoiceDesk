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


def test_session_limiter_daily_and_per_ip_caps():
    from server.app import SessionLimiter

    lim = SessionLimiter(daily_cap=3, per_ip_hour=2)
    t = 1_000_000.0
    assert lim.check("a", t) is None and lim.check("a", t + 1) is None
    assert "several calls" in (lim.check("a", t + 2) or "")
    assert lim.check("a", t + 4000) is None  # hourly window slid, and this is call 3 of the day
    assert "daily demo limit" in (lim.check("b", t + 4001) or "")
    assert lim.check("b", t + 90_000) is None  # next UTC day resets the counter
