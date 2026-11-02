"""
Speech-to-text for the dialogue modalities (2.5 and 2.6).

faster-whisper large-v3-turbo on the GPU, English, greedy decoding. One job:
take a participant's spoken turn and return the words, as fast as possible.

    POST /transcribe   raw audio bytes (webm/opus from Chrome/Android, mp4/aac
                       from iOS Safari) -> {text, audioSec, noSpeech, ms}
    GET  /health       -> {ok, model, device}

Privacy (requirements §4.1): the audio is decoded in memory and never written
to disk, and the transcript is never logged. Only timings and durations are.

Replaces the earlier Gradio service on port 7005, which collided with the
IndexTTS voice engine on the same port; this runs on 127.0.0.1:7010.
"""
import io
import os
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

HOST = "127.0.0.1"
PORT = int(os.environ.get("STT_PORT", "7010"))
MODEL_NAME = os.environ.get("STT_MODEL", "large-v3-turbo")
MODEL_DIR = os.environ.get("STT_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "float16")
SAMPLE_RATE = 16000
# A turn is capped at 30 s on the client; refuse anything absurd rather than
# tie the GPU up decoding it.
MAX_BYTES = 8 * 1024 * 1024
# Above this probability-of-no-speech a segment is most likely breath or room
# noise that Whisper would otherwise hallucinate into "Thank you." or similar.
NO_SPEECH_THRESHOLD = 0.6

t0 = time.perf_counter()
model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_DIR,
    local_files_only=True,
)
# One participant at a time, but make it explicit: two overlapping turns would
# only slow each other down on the one GPU.
_lock = threading.Lock()


def _transcribe(audio: np.ndarray, vad: bool = True) -> tuple[str, bool]:
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=vad,
    )
    kept = [s for s in segments if s.no_speech_prob < NO_SPEECH_THRESHOLD]
    text = " ".join(s.text.strip() for s in kept).strip()
    return text, not text


# Warm-up: the first call pays for CUDA kernels, memory pools and loading the
# voice-activity model, so the first real turn should not. The tone goes in
# with voice detection off: with it on, a tone is dropped before Whisper sees
# it and nothing warms up, which is how the first turn after a restart came to
# take 660 ms instead of about 120.
tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
_transcribe(tone, vad=False)
_transcribe(tone, vad=False)
_transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
print(f"[STT] {MODEL_NAME} ({COMPUTE_TYPE}) ready in {time.perf_counter() - t0:.1f}s on {HOST}:{PORT}", flush=True)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": "cuda", "computeType": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty audio")
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="audio too large")

    started = time.perf_counter()
    try:
        # In memory: PyAV reads the container straight from the bytes.
        audio = decode_audio(io.BytesIO(body), sampling_rate=SAMPLE_RATE)
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode audio")
    decoded = time.perf_counter()

    with _lock:
        text, no_speech = _transcribe(audio)
    done = time.perf_counter()

    audio_sec = len(audio) / SAMPLE_RATE
    ms = int((done - started) * 1000)
    # Timings only. The words themselves are never written anywhere.
    print(f"[STT] {audio_sec:.1f}s audio -> {ms}ms (decode {int((decoded - started) * 1000)}ms)"
          f"{' no-speech' if no_speech else ''}", flush=True)
    return {"text": text, "audioSec": round(audio_sec, 2), "noSpeech": no_speech, "ms": ms}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", access_log=False)
