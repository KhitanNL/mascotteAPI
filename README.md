# mascotteAPI

A talking avatar you can chat with: a language model writes the reply, a voice
speaks it, and a face says it on screen, about 1.5 s after you send.

This repository has two parts:

- **GPU worker** (`worker/`, `docker/`): everything that needs a GPU, behind
  one small authenticated API. Built first.
- **Web app** (next): accounts, payments, the chat page, the waiting room, and
  the router that sends each call to a worker with room and starts cloud
  workers when the others are full.

## How capacity works

One RTX 5090 carries about **5 live calls at once** (measured: the face starts
within ~2 s for 2 simultaneous replies, lag appears at 4, 6 is unusable). A
call holds one face session for its whole length. The worker reports its free
slots and refuses a new call with `503 {"error":"full"}` when it has none, so
the web app can try the next worker.

```
browser ── web app ──┬── worker: office 5090        (always on, free)
                     ├── worker: cloud 5090 #1      (started when the office one is nearly full)
                     └── worker: cloud 5090 #2 ...  (shut down after 10 idle minutes)
```

## Worker API

All `/v1` routes need `Authorization: Bearer $WORKER_TOKEN`.

| Route | What |
|---|---|
| `GET /ping` | liveness, no token |
| `GET /v1/status` | engines up, call slots (`free`), GPU |
| `POST /v1/sessions?steps=4` | body: face image → `{id, idle, poster, ...}` (opens a call slot) |
| `POST /v1/sessions/{id}/clips?turn=1&seq=0` | body: WAV of one sentence → MP4 of the face saying it |
| `POST /v1/sessions/{id}/cancel?turn=1` | stop rendering a reply that was interrupted |
| `DELETE /v1/sessions/{id}` | end the call, free the slot |
| `POST /v1/transcribe` | body: audio (webm/ogg/mp4/wav) → `{text, noSpeech}` |
| `POST /v1/chat` | Ollama `/api/chat` body → streamed NDJSON |

Voice: ElevenLabs for now (called by the web app); a local voice engine joins
the worker later.

## Run on the office machine

The engines already run there (they are shared with another project), so the
gateway just sits in front of them and loads nothing new:

```bash
cp .env.example .env        # then set WORKER_TOKEN
scripts/run-local.sh test   # gateway tests, no GPU needed
scripts/run-local.sh        # gateway on 127.0.0.1:7100
```

`MAX_SLOTS=3` there keeps 2 of the 5 calls free for the other project.

## Run on a cloud GPU

```bash
docker build -f docker/Dockerfile -t mascotte-worker .
docker run --gpus all -p 7100:7100 -e WORKER_TOKEN=... mascotte-worker
```

The image bakes in the models (~25 GB) so it is ready about a minute after
start. Push it to a registry and point RunPod (or any provider) at it.
Use a strong `WORKER_TOKEN` and, where possible, a private network between the
web app and the workers.

## Licences

- SoulX-FlashHead: Apache-2.0 (code); check the model card for the weights.
- Llama 3.1: Meta's community licence (commercial use allowed within its terms).
- faster-whisper / Whisper: MIT.
