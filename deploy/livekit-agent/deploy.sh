#!/usr/bin/env bash
# Deploy the agent to LiveKit Cloud Agents. Needs `lk` (brew install livekit-cli) and a configured
# project (`lk project add ...`). Secrets never touch the repo: pass them in the environment
# (GROQ_API_KEY, LLM_API_KEY, LLM_BASE_URL); a temporary secrets file outside the repo is created
# and removed. The first deploy uses `lk agent create`, later ones `lk agent deploy`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
SECRETS="$(mktemp)"
trap 'rm -rf "$STAGE" "$SECRETS"' EXIT

rsync -a --exclude .venv --exclude .git --exclude .env --exclude docs/screenshots \
  --exclude node_modules --exclude __pycache__ "$ROOT/" "$STAGE/"
cp "$ROOT/deploy/livekit-agent/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/deploy/livekit-agent/livekit.toml" "$STAGE/livekit.toml" 2>/dev/null || true
printf 'GROQ_API_KEY=%s\nLLM_API_KEY=%s\nLLM_BASE_URL=%s\n' \
  "$GROQ_API_KEY" "$LLM_API_KEY" "${LLM_BASE_URL:-https://api.groq.com/openai/v1}" > "$SECRETS"

cd "$STAGE"
if [ -f livekit.toml ]; then
  lk agent deploy --secrets-file "$SECRETS" "$@"
else
  lk agent create --secrets-file "$SECRETS" "$@"
  cp livekit.toml "$ROOT/deploy/livekit-agent/livekit.toml"
fi
