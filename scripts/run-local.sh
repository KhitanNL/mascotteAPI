#!/usr/bin/env bash
# Run the gateway on this machine in front of the engines that are already
# running (the office 5090). Nothing new is loaded on the GPU.
#
#   scripts/run-local.sh            # foreground
#   scripts/run-local.sh test       # run the gateway tests only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
# Install only when the requirements changed, so a restart works without internet.
REQ="$ROOT/worker/gateway/requirements.txt"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
if [ ! -f "$VENV/.req" ] || ! cmp -s "$REQ" "$VENV/.req"; then
  "$VENV/bin/pip" install -q -r "$REQ" && cp "$REQ" "$VENV/.req"
fi

if [ "${1:-}" = "test" ]; then
  cd "$ROOT/worker/gateway" && exec "$VENV/bin/python" -m pytest -q
fi

# .env holds WORKER_TOKEN (and any overrides); never committed.
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
# Shared with the research study on this machine: keep 2 of its 5 calls free for it.
export MAX_SLOTS="${MAX_SLOTS:-3}" GPU_CALLS="${GPU_CALLS:-5}" WORKER_NAME="${WORKER_NAME:-office-5090}"
cd "$ROOT/worker/gateway"
exec "$VENV/bin/python" app.py
