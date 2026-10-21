#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m booking_service &
exec python -m agent.main start
