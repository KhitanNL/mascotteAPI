#!/usr/bin/env bash
# Start the three engines, wait until each answers, then the gateway (the only open port).
set -uo pipefail
: "${WORKER_TOKEN:?set WORKER_TOKEN (24+ characters)}"
W=/opt/mascotte/worker
log() { echo "[start] $*"; }

/opt/mascotte/fetch-models.sh >/tmp/fetch.log 2>&1 || { log "model download failed"; tail -20 /tmp/fetch.log; exit 1; }

log "brain: ollama"
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=-1 ollama serve >/tmp/ollama.log 2>&1 &

log "ears: faster-whisper"
SITE=$(/opt/venv-stt/bin/python -c 'import site; print(site.getsitepackages()[0])')
LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}" HF_HUB_OFFLINE=1 \
  /opt/venv-stt/bin/python "$W/engines/stt/app.py" >/tmp/stt.log 2>&1 &

log "face: FlashHead"
/opt/venv-face/bin/python "$W/engines/face/server.py" >/tmp/face.log 2>&1 &

wait_for() {  # url name seconds
  for _ in $(seq 1 "$3"); do curl -sf "$1" >/dev/null && { log "$2 up"; return 0; }; sleep 1; done
  log "$2 did not start"; tail -30 "/tmp/$2.log" 2>/dev/null; exit 1
}
wait_for http://127.0.0.1:11434/api/tags ollama 120
# Load the model onto the GPU now, so the first caller does not wait for it.
curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"${LLM_MODEL:-llama3.1:8b}\",\"keep_alive\":-1}" >/dev/null
wait_for http://127.0.0.1:7010/health stt 300
wait_for http://127.0.0.1:7012/health face 600

log "gateway on :${WORKER_PORT:-7100}"
cd "$W/gateway"
WORKER_HOST=0.0.0.0 exec /opt/venv-gw/bin/python app.py
