#!/usr/bin/env bash
# Stage the static web client and the token app into a temp dir and deploy it to Vercel.
# Env vars (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, caps) are set once with
# `vercel env add`, never stored in files. Usage: deploy/vercel/deploy.sh [--prod]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${STAGE_DIR:-$HOME/.cache/voicedesk-vercel}"   # kept between runs so the project link persists
rm -rf "$STAGE/api"; mkdir -p "$STAGE/public"
cp "$ROOT/server/static/index.html" "$STAGE/public/index.html"
mkdir -p "$STAGE/public/static" && cp "$ROOT/server/static/app.js" "$ROOT/server/static/style.css" "$STAGE/public/static/"
cp "$ROOT/deploy/vercel/app.py" "$ROOT/server/limiter.py" "$STAGE/"
cp "$ROOT/deploy/vercel/requirements.txt" "$ROOT/deploy/vercel/vercel.json" "$STAGE/"
cd "$STAGE"
vercel deploy --yes "$@"
