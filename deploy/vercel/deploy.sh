#!/usr/bin/env bash
# Stage the static web client and the token function into a temp dir and deploy it to Vercel.
# Env vars (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, caps) are set once with
# `vercel env add`, never stored in files. Usage: deploy/vercel/deploy.sh [--prod]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${STAGE_DIR:-$HOME/.cache/voicedesk-vercel}"   # kept between runs so the project link persists
mkdir -p "$STAGE/public" "$STAGE/api"
cp "$ROOT/server/static/index.html" "$STAGE/public/index.html"
mkdir -p "$STAGE/public/static" && cp "$ROOT/server/static/app.js" "$ROOT/server/static/style.css" "$STAGE/public/static/"
cp "$ROOT"/deploy/vercel/api/*.py "$STAGE/api/"
cp "$ROOT/server/limiter.py" "$STAGE/api/limiter.py"
cp "$ROOT/deploy/vercel/requirements.txt" "$ROOT/deploy/vercel/vercel.json" "$STAGE/"
cd "$STAGE"
vercel deploy --yes "$@"
