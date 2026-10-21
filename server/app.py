"""FastAPI service: issues LiveKit access tokens, serves the web client, health and metrics."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from voicedesk.config import get_settings, load_config

from .limiter import SessionLimiter

STATIC = Path(__file__).parent / "static"

registry = CollectorRegistry()
TOKENS = Counter("voicedesk_tokens_issued_total", "LiveKit tokens issued", registry=registry)
HTTP_SECONDS = Histogram(
    "voicedesk_http_request_seconds", "HTTP request latency", ["path", "status"], registry=registry
)

app = FastAPI(title="VoiceDesk token server", version="0.1.0")


class TokenRequest(BaseModel):
    name: str = Field(default="Caller", min_length=1, max_length=40)


class TokenResponse(BaseModel):
    url: str
    token: str
    room: str
    identity: str


@app.middleware("http")
async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    HTTP_SECONDS.labels(path, str(response.status_code)).observe(time.perf_counter() - start)
    return response


limiter = SessionLimiter(
    int(os.getenv("DEMO_DAILY_SESSION_CAP", "0")), int(os.getenv("DEMO_PER_IP_HOURLY_CAP", "0"))
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/config")
def client_config() -> dict[str, str]:
    cfg = load_config()
    return {
        "clinic": cfg["clinic"]["name"],
        "note": cfg["clinic"]["note"],
        "version": cfg["version"],
    }


@app.post("/api/token", response_model=TokenResponse)
def issue_token(body: TokenRequest, request: Request) -> TokenResponse:
    s = get_settings()
    if not s.livekit_api_key or not s.livekit_api_secret:
        raise HTTPException(503, "LiveKit credentials are not configured")
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
    refusal = limiter.check(ip)
    if refusal:
        raise HTTPException(429, refusal)
    room = f"clinic-{secrets.token_hex(4)}"
    identity = f"caller-{secrets.token_hex(3)}"
    token = (
        api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
        .with_identity(identity)
        .with_name(body.name.strip() or "Caller")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    TOKENS.inc()
    return TokenResponse(
        url=s.livekit_public_url or s.livekit_url, token=token, room=room, identity=identity
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
