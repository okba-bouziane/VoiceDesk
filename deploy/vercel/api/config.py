"""Vercel Python function: public client config (mirrors /api/config of the FastAPI server)."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps(
            {
                "clinic": "Riad Sample Clinic",
                "note": "Sample data - fictional clinic",
                "version": os.getenv("PROMPT_VERSION", "2026.10.1"),
                "max_call_seconds": os.getenv("MAX_CALL_SECONDS", "150"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
