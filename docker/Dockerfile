# mascotteAPI GPU worker: face (FlashHead), ears (faster-whisper), brain (Ollama)
# and the gateway, in one image for a cloud GPU (RTX 5090 / Blackwell, CUDA 12.8).
#
#   docker build -f docker/Dockerfile -t mascotte-worker .
#   docker run --gpus all -p 7100:7100 -e WORKER_TOKEN=... mascotte-worker
#
# Models are baked in (~25 GB) so a cloud machine is ready about a minute after
# it starts, instead of downloading for ten. Build with --build-arg BAKE_MODELS=0
# to fetch them at first start instead (into /models, mount a volume there).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip git curl ca-certificates ffmpeg zstd build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ollama (the language model).
RUN curl -fsSL https://ollama.com/install.sh | sh

# Face: SoulX-FlashHead (Apache-2.0) with torch built for sm_120.
ARG FLASHHEAD_REF=main
RUN git clone --depth=1 --branch ${FLASHHEAD_REF} https://github.com/Soul-AILab/SoulX-FlashHead.git /opt/SoulX-FlashHead
RUN python3.10 -m venv /opt/venv-face \
 && /opt/venv-face/bin/pip install -q --upgrade pip wheel setuptools ninja \
 && /opt/venv-face/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
 && grep -vE '^(xformers|nvidia-nccl-cu12|decord)' /opt/SoulX-FlashHead/requirements.txt > /tmp/fh-req.txt \
 && /opt/venv-face/bin/pip install -r /tmp/fh-req.txt soundfile fastapi uvicorn huggingface_hub

# Ears: faster-whisper with its own cuBLAS/cuDNN wheels.
COPY worker/engines/stt/requirements.txt /tmp/stt-req.txt
RUN python3.10 -m venv /opt/venv-stt && /opt/venv-stt/bin/pip install -r /tmp/stt-req.txt

# Gateway.
COPY worker/gateway/requirements.txt /tmp/gw-req.txt
RUN python3.10 -m venv /opt/venv-gw && /opt/venv-gw/bin/pip install -r /tmp/gw-req.txt

COPY worker /opt/mascotte/worker
COPY docker/start.sh docker/fetch-models.sh /opt/mascotte/
RUN chmod +x /opt/mascotte/*.sh

ARG BAKE_MODELS=1
ARG LLM_MODEL=llama3.1:8b
ENV LLM_MODEL=${LLM_MODEL} OLLAMA_MODELS=/models/ollama STT_MODEL_DIR=/models/whisper
RUN if [ "$BAKE_MODELS" = "1" ]; then /opt/mascotte/fetch-models.sh; fi

EXPOSE 7100
CMD ["/opt/mascotte/start.sh"]
