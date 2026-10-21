"""Vercel Python function: issues a LiveKit access token (same contract as the FastAPI server).

Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, optional DEMO_DAILY_SESSION_CAP and
DEMO_PER_IP_HOURLY_CAP. The caps are in memory per warm instance, so they are best effort; the
authoritative daily cap and call-length limit are enforced inside the agent.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler

import jwt
from limiter import SessionLimiter

limiter = SessionLimiter(
    int(os.getenv("DEMO_DAILY_SESSION_CAP", "0")), int(os.getenv("DEMO_PER_IP_HOURLY_CAP", "0"))
)


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("content-length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            name = str(payload.get("name", "Caller")).strip()[:40] or "Caller"
        except ValueError:
            return self._send(422, {"detail": "invalid JSON"})

        url, key, secret = (
            os.getenv(k, "") for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
        )
        if not (url and key and secret):
            return self._send(503, {"detail": "LiveKit credentials are not configured"})

        ip = (self.headers.get("x-forwarded-for", "").split(",")[0].strip()) or "?"
        refusal = limiter.check(ip)
        if refusal:
            return self._send(429, {"detail": refusal})

        room, identity = f"clinic-{secrets.token_hex(4)}", f"caller-{secrets.token_hex(3)}"
        now = int(time.time())
        token = jwt.encode(
            {
                "iss": key,
                "sub": identity,
                "name": name,
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
        self._send(200, {"url": url, "token": token, "room": room, "identity": identity})
