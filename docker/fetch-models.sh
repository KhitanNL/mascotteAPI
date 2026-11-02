#!/usr/bin/env bash
# Download every model the worker needs. Idempotent: skips what is already there.
set -euo pipefail
FH=/opt/SoulX-FlashHead/models
mkdir -p "$FH" "${STT_MODEL_DIR:-/models/whisper}" "${OLLAMA_MODELS:-/models/ollama}"

echo "==> FlashHead + wav2vec2 (~15 GB)"
HF_HUB_OFFLINE=0 /opt/venv-face/bin/python - <<PY
import os
from huggingface_hub import snapshot_download
for repo, sub in (("Soul-AILab/SoulX-FlashHead-1_3B", "SoulX-FlashHead-1_3B"),
                  ("facebook/wav2vec2-base-960h", "wav2vec2-base-960h")):
    target = os.path.join("$FH", sub)
    if not os.path.isdir(target) or not os.listdir(target):
        snapshot_download(repo_id=repo, local_dir=target)
    print("   ", sub)
PY

echo "==> Whisper ${STT_MODEL:-large-v3-turbo}"
HF_HUB_OFFLINE=0 /opt/venv-stt/bin/python - <<PY
from faster_whisper import WhisperModel
WhisperModel("${STT_MODEL:-large-v3-turbo}", device="cpu", compute_type="int8", download_root="${STT_MODEL_DIR:-/models/whisper}")
PY

echo "==> ${LLM_MODEL:-llama3.1:8b}"
ollama serve >/tmp/ollama-pull.log 2>&1 &
pid=$!
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
ollama pull "${LLM_MODEL:-llama3.1:8b}"
kill "$pid" 2>/dev/null || true
echo "==> models ready"
