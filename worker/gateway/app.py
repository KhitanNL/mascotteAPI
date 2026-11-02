"""mascotteAPI GPU worker: one front door for the engines on one GPU machine.

Behind it, on the same machine:
    face     FlashHead avatar API   (AVATAR_URL, default http://127.0.0.1:7012)
    ears     faster-whisper         (STT_URL,    default http://127.0.0.1:7010)
    brain    Ollama                 (OLLAMA_URL, default http://127.0.0.1:11434)

The web app talks only to this gateway. Every worker looks the same, whether
it is the office 5090 or a cloud GPU, so adding capacity means starting
another worker and adding its URL to the web app's list.

    GET    /ping                                   no token; {ok}
    GET    /v1/status                              engines up, call slots, GPU
    GET    /v1/faces                               sample portraits: {faces: [{id, url}]}
    GET    /v1/faces/{id}                          one sample portrait (JPEG/PNG)
    GET    /v1/settings                            the face settings, with types, ranges and defaults
    POST   /v1/sessions?quality&variant&autoFrame&idleLoop  body: face image -> {id, idle, poster, settings, ...}
    POST   /v1/sessions?face={id}&...              the same with a sample portrait, no body
                                                   503 {"error": "full"} when no slot is free
    POST   /v1/sessions/{id}/clips?turn&seq        body: WAV -> video/mp4 (+ X-* timings)
    POST   /v1/sessions/{id}/cancel?turn
    DELETE /v1/sessions/{id}
    POST   /v1/transcribe                          body: audio -> {text, noSpeech, ...}
    POST   /v1/chat                                Ollama /api/chat body, streamed back as NDJSON

Every /v1 route needs "Authorization: Bearer $WORKER_TOKEN".

Call slots. A live call holds one face session for its whole length, and one
GPU carries about GPU_CALLS of them at once (5 measured on an RTX 5090). The
gateway refuses a new call when either
  - its own calls reach MAX_SLOTS (on a shared machine, set this below
    GPU_CALLS so other users of the GPU keep room), or
  - the face engine as a whole, counting sessions this gateway did not open,
    reaches GPU_CALLS.
A refusal is quick and explicit (503 "full"), so the web app can try the next
worker or start a cloud one.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


@dataclass
class Config:
    token: str
    avatar_url: str = "http://127.0.0.1:7012"
    stt_url: str = "http://127.0.0.1:7010"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.1:8b"
    gpu_calls: int = 5
    max_slots: int = 5
    # The face engine drops a session after 10 idle minutes; forget ours a bit later.
    session_idle_sec: int = 11 * 60
    worker_name: str = "worker"
    # Sample portraits offered to users (not in git: rights stay with their owner).
    faces_dir: str = str(Path(__file__).resolve().parents[2] / "faces")

    @staticmethod
    def from_env() -> "Config":
        token = os.environ.get("WORKER_TOKEN", "")
        if len(token) < 24:
            raise SystemExit("WORKER_TOKEN must be set (24+ characters): the gateway is the worker's only lock.")
        gpu_calls = int(os.environ.get("GPU_CALLS", "5"))
        return Config(
            token=token,
            avatar_url=os.environ.get("AVATAR_URL", Config.avatar_url).rstrip("/"),
            stt_url=os.environ.get("STT_URL", Config.stt_url).rstrip("/"),
            ollama_url=os.environ.get("OLLAMA_URL", Config.ollama_url).rstrip("/"),
            llm_model=os.environ.get("LLM_MODEL", Config.llm_model),
            gpu_calls=gpu_calls,
            max_slots=min(gpu_calls, int(os.environ.get("MAX_SLOTS", str(gpu_calls)))),
            worker_name=os.environ.get("WORKER_NAME", os.uname().nodename),
            faces_dir=os.environ.get("FACES_DIR", Config.faces_dir),
        )


@dataclass
class Slots:
    """The face sessions this gateway opened, with when each was last used."""
    last_used: dict[str, float] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self, sid: str) -> None:
        if sid in self.last_used:
            self.last_used[sid] = time.monotonic()

    def sweep(self, idle_sec: int) -> None:
        cutoff = time.monotonic() - idle_sec
        for sid in [s for s, t in self.last_used.items() if t < cutoff]:
            self.last_used.pop(sid, None)


# Face settings a caller may choose when opening a face. GET /v1/settings returns
# this, so a client can build its controls from it. Each maps to a FlashHead
# engine parameter; the engine's own names are accepted too (steps, seed, crop,
# idle_chunks).
SECONDS_PER_IDLE_CHUNK = 1.32  # 33 frames at 25 fps
SETTINGS = {
    "quality": {
        "type": "choice", "options": ["best", "fast"], "default": "best",
        "description": "best: 4 render steps. fast: 2 steps, about twice as fast, slightly softer face.",
    },
    "variant": {
        "type": "integer", "min": 0, "max": 2_147_483_647, "default": 42,
        "description": "Picks a variant of the head's movement style for the same face.",
    },
    "autoFrame": {
        "type": "boolean", "default": True,
        "description": "Find and frame the face in the photo. Turn off only for a tight, centred portrait.",
    },
    "idleLoop": {
        "type": "integer", "min": 1, "max": 6, "default": 3, "unit": f"× {SECONDS_PER_IDLE_CHUNK} s",
        "description": "Length of the silent rest loop between replies. Longer looks less repetitive, opens a little slower.",
    },
}


class SettingsError(ValueError):
    pass


def resolve_settings(params: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """Query parameters -> (the effective settings, the engine's parameters). Raises SettingsError."""
    def pick(name: str, engine_name: str) -> str | None:
        return params.get(name, params.get(engine_name))

    out = {k: v["default"] for k, v in SETTINGS.items()}
    q = pick("quality", "steps")
    if q is not None:
        if q in ("best", "4"):
            out["quality"] = "best"
        elif q in ("fast", "2"):
            out["quality"] = "fast"
        else:
            raise SettingsError("quality must be best or fast (steps 4 or 2)")
    v = pick("variant", "seed")
    if v is not None:
        if not v.isdigit() or int(v) > SETTINGS["variant"]["max"]:
            raise SettingsError("variant must be a whole number from 0")
        out["variant"] = int(v)
    a = pick("autoFrame", "crop")
    if a is not None:
        if a.lower() not in ("1", "0", "true", "false"):
            raise SettingsError("autoFrame must be true or false")
        out["autoFrame"] = a.lower() in ("1", "true")
    i = pick("idleLoop", "idle_chunks")
    if i is not None:
        if not i.isdigit() or not 1 <= int(i) <= 6:
            raise SettingsError("idleLoop must be 1 to 6")
        out["idleLoop"] = int(i)
    engine = {
        "steps": "4" if out["quality"] == "best" else "2",
        "seed": str(out["variant"]),
        "crop": "1" if out["autoFrame"] else "0",
        "idle_chunks": str(out["idleLoop"]),
    }
    return out, engine


def _gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def create_app(cfg: Config, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    slots = Slots()
    # Clips wait their turn for the GPU, so allow minutes, not seconds.
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(300.0, connect=5.0))

    def auth(request: Request) -> None:
        header = request.headers.get("authorization", "")
        given = header[7:] if header.lower().startswith("bearer ") else ""
        if not given or not secrets.compare_digest(given, cfg.token):
            raise HTTPException(401, "bad or missing token")

    async def engine_sessions() -> int | None:
        try:
            r = await client.get(f"{cfg.avatar_url}/health", timeout=5.0)
            return int(r.json().get("sessions", 0)) if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def up(url: str) -> bool:
        try:
            return (await client.get(url, timeout=5.0)).status_code == 200
        except httpx.HTTPError:
            return False

    def passthrough(r: httpx.Response, media_type: str | None = None) -> Response:
        headers = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}
        return Response(content=r.content, status_code=r.status_code,
                        media_type=media_type or r.headers.get("content-type"), headers=headers)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/v1/status", dependencies=[Depends(auth)])
    async def status():
        slots.sweep(cfg.session_idle_sec)
        engine, stt, llm = await asyncio.gather(
            engine_sessions(), up(f"{cfg.stt_url}/health"), up(f"{cfg.ollama_url}/api/tags"))
        mine = len(slots.last_used)
        free = 0 if engine is None else max(0, min(cfg.max_slots - mine, cfg.gpu_calls - engine))
        return {
            "worker": cfg.worker_name,
            "ready": engine is not None and stt and llm,
            "engines": {"face": engine is not None, "stt": stt, "llm": llm},
            "slots": {"gpuCalls": cfg.gpu_calls, "maxSlots": cfg.max_slots, "mine": mine,
                      "engineSessions": engine, "free": free},
            "llmModel": cfg.llm_model,
            "gpu": _gpu_name(),
        }

    def sample_faces() -> dict[str, Path]:
        d = Path(cfg.faces_dir)
        if not d.is_dir():
            return {}
        return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    @app.get("/v1/faces", dependencies=[Depends(auth)])
    async def list_faces():
        return {"faces": [{"id": k, "url": f"/v1/faces/{k}"} for k in sample_faces()]}

    @app.get("/v1/faces/{face_id}", dependencies=[Depends(auth)])
    async def get_face(face_id: str):
        p = sample_faces().get(face_id)
        if not p:
            raise HTTPException(404, "no such face")
        return Response(content=p.read_bytes(), media_type="image/png" if p.suffix.lower() == ".png" else "image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/v1/settings", dependencies=[Depends(auth)])
    async def settings():
        return {"settings": SETTINGS}

    @app.post("/v1/sessions", dependencies=[Depends(auth)])
    async def open_session(request: Request):
        query = dict(request.query_params)
        sample = query.pop("face", None)
        try:
            chosen, params = resolve_settings(query)
        except SettingsError as exc:
            return JSONResponse({"error": "bad_setting", "message": str(exc)}, status_code=400)
        if sample:
            # A sample face by id instead of an upload.
            p = sample_faces().get(sample)
            if not p:
                raise HTTPException(404, "no such face")
            body, ctype = p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        else:
            body, ctype = await request.body(), request.headers.get("content-type", "application/octet-stream")
        if not body:
            raise HTTPException(400, "face image missing")
        # One at a time, so two calls cannot both take the last slot.
        async with slots.lock:
            slots.sweep(cfg.session_idle_sec)
            engine = await engine_sessions()
            if engine is None:
                return JSONResponse({"error": "face_engine_down"}, status_code=503)
            if len(slots.last_used) >= cfg.max_slots or engine >= cfg.gpu_calls:
                return JSONResponse({"error": "full", "mine": len(slots.last_used), "engineSessions": engine},
                                    status_code=503)
            r = await client.post(f"{cfg.avatar_url}/sessions", params=params, content=body,
                                  headers={"content-type": ctype})
            if r.status_code == 200:
                opened = r.json()
                sid = opened.get("id")
                if sid:
                    slots.last_used[sid] = time.monotonic()
                return JSONResponse({**opened, "settings": chosen})
        return passthrough(r, "application/json")

    @app.post("/v1/sessions/{sid}/clips", dependencies=[Depends(auth)])
    async def clip(sid: str, request: Request):
        slots.touch(sid)
        r = await client.post(f"{cfg.avatar_url}/sessions/{sid}/clips", params=dict(request.query_params),
                              content=await request.body(), headers={"content-type": "audio/wav"})
        if r.status_code == 404:
            slots.last_used.pop(sid, None)
        slots.touch(sid)
        return passthrough(r)

    @app.post("/v1/sessions/{sid}/cancel", dependencies=[Depends(auth)])
    async def cancel(sid: str, request: Request):
        r = await client.post(f"{cfg.avatar_url}/sessions/{sid}/cancel", params=dict(request.query_params))
        return passthrough(r, "application/json")

    @app.delete("/v1/sessions/{sid}", dependencies=[Depends(auth)])
    async def close_session(sid: str):
        slots.last_used.pop(sid, None)
        r = await client.delete(f"{cfg.avatar_url}/sessions/{sid}")
        return passthrough(r, "application/json")

    @app.post("/v1/transcribe", dependencies=[Depends(auth)])
    async def transcribe(request: Request):
        r = await client.post(f"{cfg.stt_url}/transcribe", params=dict(request.query_params),
                              content=await request.body(),
                              headers={"content-type": request.headers.get("content-type", "application/octet-stream")})
        return passthrough(r, "application/json")

    @app.post("/v1/chat", dependencies=[Depends(auth)])
    async def chat(request: Request):
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(400, "JSON body expected")
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise HTTPException(400, "messages missing")
        payload.setdefault("model", cfg.llm_model)
        payload.setdefault("stream", True)
        req = client.build_request("POST", f"{cfg.ollama_url}/api/chat", json=payload)
        r = await client.send(req, stream=True)
        if r.status_code != 200:
            content = await r.aread()
            await r.aclose()
            return Response(content=content, status_code=r.status_code, media_type="application/json")

        async def relay():
            try:
                async for chunk in r.aiter_raw():
                    yield chunk
            finally:
                await r.aclose()

        return StreamingResponse(relay(), media_type="application/x-ndjson")

    @app.on_event("shutdown")
    async def _close():
        await client.aclose()

    app.state.slots = slots
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(Config.from_env()), host=os.environ.get("WORKER_HOST", "127.0.0.1"),
                port=int(os.environ.get("WORKER_PORT", "7100")), log_level="warning", access_log=False)
