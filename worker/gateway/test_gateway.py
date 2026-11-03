"""Gateway tests against fake engines (no GPU needed): pytest worker/gateway"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import Config, create_app

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeEngines:
    """The face engine, speech-to-text and Ollama, in memory."""

    def __init__(self, other_sessions: int = 0):
        self.sessions: set[str] = {f"other-{i}" for i in range(other_sessions)}
        self.n = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        path, host = req.url.path, req.url.host
        if host == "avatar":
            if path == "/health":
                return httpx.Response(200, json={"ok": True, "sessions": len(self.sessions), "busy": False})
            if path == "/sessions" and req.method == "POST":
                self.n += 1
                sid = f"s{self.n}"
                self.sessions.add(sid)
                return httpx.Response(200, json={"id": sid, "idle": "AAAA", "poster": "BBBB"})
            if path.endswith("/clips"):
                sid = path.split("/")[2]
                if sid not in self.sessions:
                    return httpx.Response(404, json={"detail": "no such session"})
                return httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4", "X-Render-Ms": "321"})
            if req.method == "DELETE":
                self.sessions.discard(path.split("/")[2])
                return httpx.Response(200, json={"ok": True, "deleted": True})
            if path.endswith("/cancel"):
                return httpx.Response(200, json={"ok": True})
        if host == "stt":
            if path == "/health":
                return httpx.Response(200, json={"ok": True})
            if path == "/transcribe":
                return httpx.Response(200, json={"text": "hello", "noSpeech": False})
        if host == "llm":
            if path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            if path == "/api/chat":
                body = json.loads(req.content)
                lines = [{"message": {"content": "Hi"}, "model": body["model"]}, {"done": True}]
                # A real stream, as Ollama sends, not a body read up front.
                return httpx.Response(200, stream=httpx.ByteStream("\n".join(json.dumps(l) for l in lines).encode()))
        return httpx.Response(404)


def client(engines: FakeEngines, **cfg) -> TestClient:
    config = Config(token=TOKEN, avatar_url="http://avatar", stt_url="http://stt", ollama_url="http://llm", **cfg)
    return TestClient(create_app(config, transport=httpx.MockTransport(engines.handler)))


def test_needs_the_token():
    c = client(FakeEngines())
    assert c.get("/ping").status_code == 200
    assert c.get("/v1/status").status_code == 401
    assert c.get("/v1/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/v1/status", headers=AUTH).status_code == 200


def test_status_reports_free_slots():
    s = client(FakeEngines(other_sessions=1), gpu_calls=5, max_slots=3).get("/v1/status", headers=AUTH).json()
    assert s["ready"] is True
    assert s["slots"] == {"gpuCalls": 5, "maxSlots": 3, "mine": 0, "engineSessions": 1, "free": 3}


def test_refuses_past_its_own_limit():
    c = client(FakeEngines(), gpu_calls=5, max_slots=2)
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 200
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 200
    r = c.post("/v1/sessions", content=b"jpg", headers=AUTH)
    assert r.status_code == 503 and r.json()["error"] == "full"


def test_counts_sessions_it_did_not_open():
    # Another user of the GPU (the study) holds 4 of 5: one call left for us.
    c = client(FakeEngines(other_sessions=4), gpu_calls=5, max_slots=3)
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 200
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).json()["error"] == "full"


def test_closing_frees_the_slot():
    c = client(FakeEngines(), gpu_calls=5, max_slots=1)
    sid = c.post("/v1/sessions", content=b"jpg", headers=AUTH).json()["id"]
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 503
    assert c.delete(f"/v1/sessions/{sid}", headers=AUTH).status_code == 200
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 200


def test_clip_passes_video_and_timings_through():
    c = client(FakeEngines())
    sid = c.post("/v1/sessions", content=b"jpg", headers=AUTH).json()["id"]
    r = c.post(f"/v1/sessions/{sid}/clips?turn=1&seq=0", content=b"wav", headers=AUTH)
    assert r.status_code == 200 and r.content == b"mp4"
    assert r.headers["content-type"] == "video/mp4" and r.headers["x-render-ms"] == "321"


def test_a_session_the_engine_dropped_frees_its_slot():
    engines = FakeEngines()
    c = client(engines, gpu_calls=5, max_slots=1)
    sid = c.post("/v1/sessions", content=b"jpg", headers=AUTH).json()["id"]
    engines.sessions.discard(sid)  # evicted by the engine's idle timer
    assert c.post(f"/v1/sessions/{sid}/clips?turn=1&seq=0", content=b"wav", headers=AUTH).status_code == 404
    assert c.post("/v1/sessions", content=b"jpg", headers=AUTH).status_code == 200


def test_transcribe_and_chat():
    c = client(FakeEngines())
    assert c.post("/v1/transcribe", content=b"webm", headers=AUTH).json()["text"] == "hello"
    r = c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}, headers=AUTH)
    lines = [json.loads(l) for l in r.text.splitlines() if l]
    assert lines[0]["message"]["content"] == "Hi" and lines[0]["model"] == "llama3.1:8b"
    assert c.post("/v1/chat", json={"nope": 1}, headers=AUTH).status_code == 400


def test_sample_faces(tmp_path):
    (tmp_path / "1.jpg").write_bytes(b"\xff\xd8one")
    (tmp_path / "notes.txt").write_text("not a face")
    c = client(FakeEngines(), faces_dir=str(tmp_path))
    assert c.get("/v1/faces").status_code == 401
    assert c.get("/v1/faces", headers=AUTH).json() == {"faces": [{"id": "1", "url": "/v1/faces/1"}]}
    r = c.get("/v1/faces/1", headers=AUTH)
    assert r.content == b"\xff\xd8one" and r.headers["content-type"] == "image/jpeg"
    assert c.get("/v1/faces/notes", headers=AUTH).status_code == 404
    assert c.post("/v1/sessions?face=1", headers=AUTH).json()["id"] == "s1"
    assert c.post("/v1/sessions?face=nope", headers=AUTH).status_code == 404


class RecordingEngines(FakeEngines):
    def __init__(self):
        super().__init__()
        self.opened_with: list[dict] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.host == "avatar" and req.url.path == "/sessions":
            self.opened_with.append(dict(req.url.params))
        return super().handler(req)


def test_settings_are_listed():
    s = client(FakeEngines()).get("/v1/settings", headers=AUTH).json()["settings"]
    assert set(s) == {"quality", "variant", "autoFrame", "resolution", "idleLoop"}
    assert s["resolution"]["options"] == [512, 640, 768, 1024] and s["resolution"]["default"] == 512
    assert s["quality"]["default"] == "best" and s["idleLoop"]["max"] == 6


def test_settings_map_to_the_engine():
    e = RecordingEngines()
    c = client(e)
    r = c.post("/v1/sessions?quality=fast&variant=7&autoFrame=false&resolution=768&idleLoop=5", content=b"jpg", headers=AUTH).json()
    assert r["settings"] == {"quality": "fast", "variant": 7, "autoFrame": False, "resolution": 768, "idleLoop": 5}
    assert e.opened_with[-1] == {"steps": "2", "seed": "7", "crop": "0", "idle_chunks": "5", "res": "768"}
    # Defaults, and the engine's own names still work (mascotte.ai sends steps=4).
    assert c.post("/v1/sessions?steps=4", content=b"jpg", headers=AUTH).json()["settings"] == \
        {"quality": "best", "variant": 42, "autoFrame": True, "resolution": 512, "idleLoop": 3}
    assert e.opened_with[-1] == {"steps": "4", "seed": "42", "crop": "1", "idle_chunks": "3", "res": "512"}
    # the engine's own name for it too
    c.post("/v1/sessions?res=1024", content=b"jpg", headers=AUTH)
    assert e.opened_with[-1]["res"] == "1024"


def test_bad_settings_are_refused_before_the_gpu():
    e = RecordingEngines()
    c = client(e)
    for q in ("quality=ultra", "steps=3", "variant=-1", "variant=abc", "autoFrame=maybe", "idleLoop=0", "idleLoop=9",
              "resolution=700", "res=abc", "resolution=2048"):
        r = c.post(f"/v1/sessions?{q}", content=b"jpg", headers=AUTH)
        assert r.status_code == 400 and r.json()["error"] == "bad_setting", q
    assert e.opened_with == []


def test_refuses_to_start_without_a_strong_token(monkeypatch):
    monkeypatch.setenv("WORKER_TOKEN", "short")
    with pytest.raises(SystemExit):
        Config.from_env()
