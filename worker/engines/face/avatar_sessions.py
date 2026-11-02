"""Modality 2.6: live avatar sessions on the FlashHead model.

2.4 renders one finished video from a face and a whole script. A live call
needs the opposite: the face prepared once, then each sentence of a reply
turned into a short clip as soon as its voice exists, with the head carrying
on smoothly from one clip to the next. This module adds that, as a small HTTP
API on AVATAR_PORT (7012), inside the talking-head process so it shares the
model already loaded for 2.4:

    GET    /health                              {ok, sessions, busy}
    POST   /sessions?steps&seed&crop&idle_chunks  body: image -> {id, idle, poster, ...}
    POST   /sessions/{id}/clips?turn&seq        body: WAV   -> video/mp4
    POST   /sessions/{id}/cancel?turn           stop rendering that reply
    DELETE /sessions/{id}

How sessions stay out of 2.4's way (and 2.4 out of theirs). The pipeline keeps
everything about the current face and motion as plain attributes, and
FlashHeadPipeline is an ordinary class, so copy.copy() gives each session a
"view": its own attributes (reference latent, colour reference, generator,
motion latents, timesteps), sharing the model, VAE and wav2vec2 weights by
reference. 2.4 keeps using the shared object and re-prepares it per video as
it always has; a session never touches it, and never writes the module-level
infer_params that 2.4 changes. All GPU work, both kinds, runs under the one
lock app.py already has.

Continuity. Within a reply, a clip starts exactly where the previous one ended:
the motion latents and the 8 s audio window carry over, so consecutive clips
are consecutive frames of one stream. Each new reply starts from the rest pose,
which is also where the idle loop is rendered from, so the browser's switch
from idle to speech is a small step it can cross-fade.

Privacy (requirements 4.1): the face and the audio exist in GPU memory, RAM
and, for the few milliseconds ffmpeg needs them, in /dev/shm, which is RAM.
Nothing is written to disk, nothing is logged but timings and sizes, and a
session is deleted when the call ends, with a timeout as a backstop.
"""

import asyncio
import base64
import copy
import glob
import io
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np
import soundfile as sf
import soxr
import torch
from PIL import Image

from flash_head.inference import get_audio_embedding, get_infer_params, run_pipeline

SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
TEMP_PREFIX = "fh-avatar-"
RESOLUTION = 512
FPS = 25
# Generous: the team tests how many calls the GPU can carry (renders queue on _gpu_lock).
MAX_SESSIONS = 32
SESSION_IDLE_SEC = 10 * 60
SESSION_MAX_SEC = 25 * 60
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_CLIP_SEC = 30

# Set by start(). (A string annotation: threading.Lock is a factory function,
# not a type, so "threading.Lock | None" cannot be evaluated on Python 3.10.)
_gpu_lock: "threading.Lock | None" = None
_load_pipeline = None

_sessions: dict[str, "AvatarSession"] = {}
_registry_lock = threading.Lock()


class Superseded(Exception):
    """A newer reply started, or this one was cancelled: stop rendering it."""


class Closed(Exception):
    """The session was deleted while the request was waiting."""


class AvatarSession:
    def __init__(self, sid: str, view, samples: int):
        self.id = sid
        self.view = view
        self.window = np.zeros(samples, dtype=np.float32)
        # Turns are replies, numbered by Node from 1. A request for an older
        # turn than the latest one seen is stale (the participant interrupted).
        self.latest_turn = 0
        self.rendered_turn = 0
        self.next_seq = 0
        self.closed = False
        self.created = time.monotonic()
        self.last_used = self.created

    def close(self):
        self.closed = True
        self.view = None
        self.window = None


# ── Pipeline views ──────────────────────────────────────────────────────────

def _params():
    p = get_infer_params()
    slice_frames = p["frame_num"] - p["motion_frames_num"]
    return {
        "p": p,
        "slice_frames": slice_frames,
        "slice_samples": slice_frames * p["sample_rate"] // p["tgt_fps"],
        "window_samples": p["sample_rate"] * p["cached_audio_duration"],
        "end_idx": p["cached_audio_duration"] * p["tgt_fps"],
        "start_idx": p["cached_audio_duration"] * p["tgt_fps"] - p["frame_num"],
    }


def _new_view(image_path: str, steps: int, seed: int, crop: bool):
    """A per-session copy of the pipeline. Caller holds the GPU lock."""
    base = _load_pipeline()
    view = copy.copy(base)  # own attribute dict, shared weights
    view.rank = 1           # keeps the upstream per-step prints out of the log
    p = get_infer_params()
    # Explicit arguments, never get_base_data(): that reads the module-level
    # infer_params, whose step count and size 2.4 changes per request.
    view.prepare_params(
        cond_image_path_or_dir=image_path,
        target_size=(RESOLUTION, RESOLUTION),
        frame_num=p["frame_num"],
        motion_frames_num=p["motion_frames_num"],
        sampling_steps=steps,
        seed=seed,
        shift=p["sample_shift"],
        color_correction_strength=p["color_correction_strength"],
        use_face_crop=crop,
    )
    return view


def _render_chunk(view, window: np.ndarray, k: dict) -> np.ndarray:
    """One chunk: 24 new frames as uint8 (24, H, W, 3) on the CPU. Caller holds the lock."""
    embedding = get_audio_embedding(view, window, k["start_idx"], k["end_idx"])
    frames = run_pipeline(view, embedding)[k["p"]["motion_frames_num"]:]
    return frames.to(torch.uint8).cpu().numpy()


# ── Encoding (system ffmpeg, files in /dev/shm) ─────────────────────────────

def _shm_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=suffix, dir=SHM_DIR)
    os.close(fd)
    return path


def _unlink(*paths):
    for path in paths:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


VIDEO_ARGS = [
    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-c:v", "libx264", "-preset", "veryfast",
    # One keyframe and no B-frames: each clip starts instantly and the browser
    # can swap between clips without waiting for a decoder to catch up.
    "-g", "250", "-keyint_min", "250", "-sc_threshold", "0", "-bf", "0",
    "-profile:v", "high", "-level:v", "3.1", "-threads", "4",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
]


class Encoder:
    """ffmpeg fed with raw frames from a writer thread, so encoding overlaps rendering."""

    def __init__(self, out_path: str, duration_sec: float, audio_path: str | None, crf: int, maxrate: str):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{RESOLUTION}x{RESOLUTION}",
            "-r", str(FPS), "-i", "pipe:0",
        ]
        if audio_path:
            cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0"]
        cmd += VIDEO_ARGS + ["-crf", str(crf), "-maxrate", maxrate, "-bufsize", "1200k"]
        if audio_path:
            # The voice at its own sample rate, padded with silence to the
            # video's length. -cutoff for the same reason as 2.4: ffmpeg's AAC
            # encoder otherwise low-passes the voice to near telephone quality.
            cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "96k", "-ac", "1", "-cutoff", "20000"]
        else:
            cmd += ["-an"]
        cmd += ["-t", f"{duration_sec:.3f}", "-movflags", "+faststart", out_path]
        self.out_path = out_path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames: "queue.Queue" = queue.Queue(maxsize=4)
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while True:
            item = self.frames.get()
            if item is None:
                return
            try:
                self.proc.stdin.write(item.tobytes())
            except BaseException as exc:
                self.errors.append(exc)
                while self.frames.get() is not None:
                    pass
                return

    def put(self, frames: np.ndarray):
        self.frames.put(frames)

    def finish(self) -> bytes:
        self.frames.put(None)
        self.thread.join(timeout=30)
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        code = self.proc.wait(timeout=30)
        if code != 0 or self.errors:
            err = self.proc.stderr.read().decode(errors="ignore")[:300]
            raise RuntimeError(f"ffmpeg failed ({code}): {err or self.errors}")
        with open(self.out_path, "rb") as fh:
            return fh.read()

    def abort(self):
        try:
            self.frames.put_nowait(None)
        except queue.Full:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def _poster_jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=85)
    return buf.getvalue()


# ── Session lifecycle ───────────────────────────────────────────────────────

def _register(session: AvatarSession):
    with _registry_lock:
        _sessions[session.id] = session
        while len(_sessions) > MAX_SESSIONS:
            oldest = min(_sessions.values(), key=lambda s: s.last_used)
            _sessions.pop(oldest.id, None)
            oldest.close()
            print(f"[Avatar] session evicted (limit {MAX_SESSIONS})", flush=True)


def _get(sid: str) -> AvatarSession | None:
    with _registry_lock:
        s = _sessions.get(sid)
    return s if s and not s.closed else None


def _delete(sid: str) -> bool:
    with _registry_lock:
        s = _sessions.pop(sid, None)
    if s:
        s.close()
    return s is not None


def create_session(image_bytes: bytes, steps: int, seed: int, crop: bool, idle_chunks: int) -> dict:
    k = _params()
    image_path = None
    t0 = time.monotonic()
    try:
        # Validate, flatten to RGB and hand prepare_params a path, which is
        # what it takes. The file lives in RAM for the few ms it is needed.
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
        image_path = _shm_path(".png")
        image.save(image_path, "PNG")

        idle_frames: list[np.ndarray] = []
        with _gpu_lock:
            t_lock = time.monotonic()
            view = _new_view(image_path, steps, seed, crop)
            t_prep = time.monotonic()
            # The idle loop: exact digital silence. wav2vec2 normalises over the
            # whole window, so any noise, however faint, would be amplified
            # into mouth movement; zeros stay zeros.
            window = np.zeros(k["window_samples"], dtype=np.float32)
            for _ in range(idle_chunks):
                idle_frames.append(_render_chunk(view, window, k))
            # Back to the rest pose: every reply starts from here too.
            view.reset_person_name()
            t_idle = time.monotonic()
    finally:
        _unlink(image_path)

    # Forward then backward, so the loop has no seam: 72 frames become 142.
    frames = np.concatenate(idle_frames)
    loop = np.concatenate([frames, frames[-2:0:-1]])
    idle_path = _shm_path(".mp4")
    try:
        enc = Encoder(idle_path, len(loop) / FPS, None, crf=26, maxrate="400k")
        for i in range(0, len(loop), 24):
            enc.put(loop[i:i + 24])
        idle_mp4 = enc.finish()
    finally:
        _unlink(idle_path)
    t_enc = time.monotonic()

    session = AvatarSession(uuid.uuid4().hex, view, k["window_samples"])
    _register(session)
    meta = {
        "id": session.id,
        "idle": base64.b64encode(idle_mp4).decode("ascii"),
        "poster": base64.b64encode(_poster_jpeg(frames[0])).decode("ascii"),
        "idleSec": round(len(loop) / FPS, 2),
        "waitMs": int((t_lock - t0) * 1000),
        "prepMs": int((t_prep - t_lock) * 1000),
        "idleRenderMs": int((t_idle - t_prep) * 1000),
        "encodeMs": int((t_enc - t_idle) * 1000),
        "idleKb": round(len(idle_mp4) / 1024),
    }
    print(f"[Avatar] session created: wait {meta['waitMs']} ms, prepare {meta['prepMs']} ms, "
          f"idle {meta['idleRenderMs']} ms, encode {meta['encodeMs']} ms, {meta['idleKb']} KB", flush=True)
    return meta


def render_clip(session: AvatarSession, wav_bytes: bytes, turn: int, seq: int) -> tuple[bytes, dict]:
    k = _params()
    sr_model = k["p"]["sample_rate"]
    slice_samples = k["slice_samples"]
    t0 = time.monotonic()

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    spoken_sec = len(audio) / sr
    if spoken_sec > MAX_CLIP_SEC:
        raise ValueError(f"clip too long ({spoken_sec:.1f} s)")
    a16 = soxr.resample(audio, sr, sr_model, "HQ") if sr != sr_model else audio
    chunks = max(1, math.ceil(len(a16) / slice_samples))
    a16 = np.pad(a16, (0, chunks * slice_samples - len(a16))).astype(np.float32)
    duration = chunks * k["slice_frames"] / FPS

    # The encoder starts before the GPU lock, so its start-up overlaps the wait.
    wav_path = _shm_path(".wav")
    out_path = _shm_path(".mp4")
    with open(wav_path, "wb") as fh:
        fh.write(wav_bytes)
    enc = Encoder(out_path, duration, wav_path, crf=24, maxrate="600k")
    ok = False
    try:
        with _gpu_lock:
            t_lock = time.monotonic()
            if session.closed:
                raise Closed()
            if turn < session.latest_turn:
                raise Superseded()
            if turn > session.rendered_turn:
                # A new reply starts from the rest pose, like the idle loop.
                session.view.reset_person_name()
                session.window = np.zeros_like(session.window)
                session.rendered_turn = turn
                session.next_seq = 0
            if seq < session.next_seq:
                raise Superseded()
            try:
                for i in range(chunks):
                    if session.closed:
                        raise Closed()
                    if session.latest_turn != turn:
                        raise Superseded()
                    chunk = a16[i * slice_samples:(i + 1) * slice_samples]
                    session.window = np.concatenate((session.window[slice_samples:], chunk))
                    enc.put(_render_chunk(session.view, session.window, k))
            finally:
                session.next_seq = seq + 1
                session.last_used = time.monotonic()
            t_render = time.monotonic()
        mp4 = enc.finish()
        ok = True
    finally:
        if not ok:
            enc.abort()
        _unlink(wav_path, out_path)
    t_enc = time.monotonic()

    meta = {
        "durationSec": round(duration, 2),
        "spokenSec": round(spoken_sec, 2),
        "chunks": chunks,
        "waitMs": int((t_lock - t0) * 1000),
        "renderMs": int((t_render - t_lock) * 1000),
        "encodeMs": int((t_enc - t_render) * 1000),
        "kb": round(len(mp4) / 1024),
    }
    print(f"[Avatar] clip turn {turn} seq {seq}: {meta['spokenSec']} s voice, {chunks} chunks, "
          f"wait {meta['waitMs']} ms, render {meta['renderMs']} ms, encode {meta['encodeMs']} ms, "
          f"{meta['kb']} KB", flush=True)
    return mp4, meta


def _sweep_forever():
    while True:
        time.sleep(30)
        now = time.monotonic()
        with _registry_lock:
            expired = [s for s in _sessions.values()
                       if now - s.last_used > SESSION_IDLE_SEC or now - s.created > SESSION_MAX_SEC]
        for s in expired:
            _delete(s.id)
            print("[Avatar] session expired", flush=True)


# ── Self-check ──────────────────────────────────────────────────────────────

PER_FACE_ATTRS = ("generator", "timesteps", "cond_image_dict", "cond_image_tensor_dict",
                  "ref_img_latent_dict", "ref_img_latent", "original_color_reference",
                  "latent_motion_frames")


def self_check(sample_image: str) -> None:
    """Prove a session cannot disturb 2.4's shared pipeline. Raises if it could.

    Upstream is a moving target. If a future version kept per-face state
    somewhere copy.copy() does not separate, sessions and 2.4 would corrupt
    each other silently, so the avatar API refuses to start instead.
    """
    k = _params()
    base = _load_pipeline()
    with _gpu_lock:
        before = {a: id(getattr(base, a, None)) for a in PER_FACE_ATTRS}
        view = _new_view(sample_image, 4, 7, True)
        window = np.zeros(k["window_samples"], dtype=np.float32)
        _render_chunk(view, window, k)
        after = {a: id(getattr(base, a, None)) for a in PER_FACE_ATTRS}
    if before != after:
        changed = [a for a in PER_FACE_ATTRS if before[a] != after[a]]
        raise RuntimeError(f"a session changed the shared pipeline: {changed}")
    shared = [a for a in PER_FACE_ATTRS if getattr(view, a, None) is getattr(base, a, None)]
    if shared:
        raise RuntimeError(f"a session shares per-face state with 2.4: {shared}")
    del view


# ── HTTP API ────────────────────────────────────────────────────────────────

def _build_app():
    from fastapi import FastAPI, HTTPException, Request, Response
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        with _registry_lock:
            count = len(_sessions)
        return {"ok": True, "sessions": count, "busy": _gpu_lock.locked()}

    @app.post("/sessions")
    async def create(request: Request, steps: int = 4, seed: int = 42, crop: int = 1, idle_chunks: int = 3):
        if steps not in (2, 4):
            raise HTTPException(400, "steps must be 2 or 4")
        body = await request.body()
        if not body or len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(400, "image missing or too large")
        try:
            return await run_in_threadpool(create_session, body, steps, seed, bool(crop), max(1, min(idle_chunks, 6)))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"could not read the image: {exc}")
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(500, f"session failed: {exc}")

    @app.post("/sessions/{sid}/clips")
    async def clip(sid: str, request: Request, turn: int, seq: int):
        session = _get(sid)
        if not session:
            raise HTTPException(404, "no such session")
        # Recorded on arrival, before waiting for the GPU, so a render of an
        # older reply stops at its next chunk.
        session.latest_turn = max(session.latest_turn, turn)
        body = await request.body()
        if not body:
            raise HTTPException(400, "audio missing")
        try:
            mp4, meta = await run_in_threadpool(render_clip, session, body, turn, seq)
        except Superseded:
            raise HTTPException(409, "superseded")
        except Closed:
            raise HTTPException(404, "session closed")
        except (ValueError, RuntimeError) as exc:  # includes soundfile's decode errors
            traceback.print_exc()
            raise HTTPException(500, f"clip failed: {exc}")
        headers = {
            "X-Duration-Sec": str(meta["durationSec"]),
            "X-Chunks": str(meta["chunks"]),
            "X-Wait-Ms": str(meta["waitMs"]),
            "X-Render-Ms": str(meta["renderMs"]),
            "X-Encode-Ms": str(meta["encodeMs"]),
        }
        return Response(content=mp4, media_type="video/mp4", headers=headers)

    @app.post("/sessions/{sid}/cancel")
    async def cancel(sid: str, turn: int):
        session = _get(sid)
        if session:
            session.latest_turn = max(session.latest_turn, turn + 1)
        return {"ok": True}

    @app.delete("/sessions/{sid}")
    async def delete(sid: str):
        return {"ok": True, "deleted": _delete(sid)}

    return app


def start(port: int, lock: threading.Lock, load_pipeline, sample_image: str | None = None) -> bool:
    """Start the API in a background thread. Returns False (and logs) if it cannot."""
    global _gpu_lock, _load_pipeline
    _gpu_lock = lock
    _load_pipeline = load_pipeline

    for stale in glob.glob(os.path.join(SHM_DIR, TEMP_PREFIX + "*")):
        _unlink(stale)

    if sample_image and os.path.exists(sample_image):
        try:
            self_check(sample_image)
            print("[Avatar] self-check passed: sessions are isolated from 2.4", flush=True)
        except Exception as exc:
            traceback.print_exc()
            print(f"[Avatar] ERROR: self-check failed, 2.6 avatar API not started: {exc}", flush=True)
            return False

    import uvicorn
    config = uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="warning",
                            access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="avatar-api", daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        # uvicorn exits its thread quietly when it cannot bind; say so loudly.
        print(f"[Avatar] ERROR: the avatar API did not start on port {port} (port in use?)", flush=True)
        return False
    threading.Thread(target=_sweep_forever, name="avatar-sweeper", daemon=True).start()
    print(f"[Avatar] 2.6 avatar API on 127.0.0.1:{port}", flush=True)
    return True
