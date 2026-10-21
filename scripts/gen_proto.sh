#!/usr/bin/env bash
# Regenerate the gRPC stubs from proto/ into voicedesk/booking/v1/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" -m grpc_tools.protoc -Iproto \
  --python_out=. --pyi_out=. --grpc_python_out=. \
  proto/voicedesk/booking/v1/booking.proto
touch voicedesk/booking/__init__.py voicedesk/booking/v1/__init__.py
