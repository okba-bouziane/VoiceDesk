"""Vercel entrypoint: the token endpoint and client config (same contract as the FastAPI
server in server/app.py), as a tiny FastAPI app. The static client is served from public/.

Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEMO_DAILY_SESSION_CAP,
DEMO_PER_IP_HOURLY_CAP, MAX_CALL_SECONDS. The caps are in memory per warm instance, so they are
best effort here; the authoritative daily cap and call-length limit are enforced in the agent.
"""

from __future__ import annotations

import os
import secrets
import time

import jwt
from fastapi import FastAPI, HTTPException, Request
from limiter import SessionLimiter
from pydantic import BaseModel, Field

app = FastAPI()
limiter = SessionLimiter(
    int(os.getenv("DEMO_DAILY_SESSION_CAP", "0")), int(os.getenv("DEMO_PER_IP_HOURLY_CAP", "0"))
)


class TokenRequest(BaseModel):
    name: str = Field(default="Caller", min_length=1, max_length=40)


@app.get("/api/config")
def config() -> dict[str, str]:
    return {
        "clinic": "Riad Sample Clinic",
        "note": "Sample data - fictional clinic",
        "version": "2026.10.1",
        "max_call_seconds": os.getenv("MAX_CALL_SECONDS", "150"),
    }


@app.post("/api/token")
def token(body: TokenRequest, request: Request) -> dict[str, str]:
    url, key, secret = (
        os.getenv(k, "") for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    if not (url and key and secret):
        raise HTTPException(503, "LiveKit credentials are not configured")
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or "?"
    refusal = limiter.check(ip)
    if refusal:
        raise HTTPException(429, refusal)
    room, identity = f"clinic-{secrets.token_hex(4)}", f"caller-{secrets.token_hex(3)}"
    now = int(time.time())
    jwt_token = jwt.encode(
        {
            "iss": key,
            "sub": identity,
            "name": body.name.strip() or "Caller",
            "nbf": now,
            "exp": now + 600,
            "video": {
                "room": room,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            },
        },
        secret,
        algorithm="HS256",
    )
    return {"url": url, "token": jwt_token, "room": room, "identity": identity}
