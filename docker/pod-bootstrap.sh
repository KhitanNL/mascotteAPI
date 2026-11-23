#!/usr/bin/env bash
# A RunPod pod without our image: installs the face engine and the gateway, then
# serves the worker API on :7100. Only what a mascotte.ai call needs (mascotte.ai
# has its own LLM and voice), so no speech-to-text and no Ollama.
#
# Everything lives on the /workspace volume: the first start installs and
# downloads (~10 min), a stopped and restarted pod only starts the engines.
#
# RunPod pod: image runpod/pytorch (any CUDA 12.8 tag), HTTP port 7100,
# volume on /workspace (40 GB), env WORKER_TOKEN, start command:
#   bash -c "(curl -fsSL https://raw.githubusercontent.com/KhitanNL/mascotteAPI/main/docker/pod-bootstrap.sh | bash > /workspace/boot.log 2>&1 &); /start.sh"
set -euo pipefail
: "${WORKER_TOKEN:?set WORKER_TOKEN in the pod environment}"
W=/workspace/mascotte
mkdir -p "$W/logs"
log() { echo "[boot $(date +%T)] $*"; }
T0=$(date +%s)

command -v ffmpeg >/dev/null || { log "apt: ffmpeg"; apt-get update -qq && apt-get install -y -qq ffmpeg git curl >/dev/null; }
command -v uv >/dev/null || { log "uv"; pip install -q uv; }

log "code"
if [ -d "$W/mascotteAPI/.git" ]; then git -C "$W/mascotteAPI" pull -q; else git clone -q --depth=1 https://github.com/KhitanNL/mascotteAPI.git "$W/mascotteAPI"; fi
[ -d "$W/SoulX-FlashHead/.git" ] || git clone -q --depth=1 https://github.com/Soul-AILab/SoulX-FlashHead.git "$W/SoulX-FlashHead"

if [ ! -x "$W/venv-face/bin/python" ]; then
  log "face venv (torch for RTX 50xx)"
  uv venv -q -p 3.10 "$W/venv-face"
  uv pip install -q -p "$W/venv-face" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  grep -vE '^(xformers|nvidia-nccl-cu12|decord)' "$W/SoulX-FlashHead/requirements.txt" > /tmp/fh-req.txt
  uv pip install -q -p "$W/venv-face" -r /tmp/fh-req.txt soundfile soxr fastapi uvicorn huggingface_hub
fi
if [ ! -x "$W/venv-gw/bin/python" ]; then
  log "gateway venv"
  uv venv -q -p 3.10 "$W/venv-gw"
  uv pip install -q -p "$W/venv-gw" -r "$W/mascotteAPI/worker/gateway/requirements.txt"
fi

log "models"
HF_HUB_ENABLE_HF_TRANSFER=0 "$W/venv-face/bin/python" - <<PY
from huggingface_hub import snapshot_download
m = "$W/SoulX-FlashHead/models"
# Lite only: the Pro weights are not used by live calls.
snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=f"{m}/SoulX-FlashHead-1_3B", ignore_patterns=["Model_Pro/*"])
snapshot_download("facebook/wav2vec2-base-960h", local_dir=f"{m}/wav2vec2-base-960h")
PY
log "installed after $(( $(date +%s) - T0 )) s"

log "face engine"
FLASHHEAD_DIR="$W/SoulX-FlashHead" nohup "$W/venv-face/bin/python" "$W/mascotteAPI/worker/engines/face/server.py" \
  > "$W/logs/face.log" 2>&1 &
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:7012/health >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:7012/health >/dev/null || { log "face engine did not start"; tail -30 "$W/logs/face.log"; exit 1; }
log "face up after $(( $(date +%s) - T0 )) s"

log "gateway on :7100"
cd "$W/mascotteAPI/worker/gateway"
WORKER_HOST=0.0.0.0 GPU_CALLS="${GPU_CALLS:-5}" MAX_SLOTS="${MAX_SLOTS:-5}" WORKER_NAME="${WORKER_NAME:-runpod-5090}" \
  FACES_DIR=/nonexistent nohup "$W/venv-gw/bin/python" app.py > "$W/logs/gateway.log" 2>&1 &
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:7100/ping >/dev/null && break; sleep 1; done
log "ready after $(( $(date +%s) - T0 )) s"
