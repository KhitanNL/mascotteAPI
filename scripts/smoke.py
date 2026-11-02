"""End-to-end check of a running worker: python scripts/smoke.py [URL]  (token from WORKER_TOKEN)

Opens one face, renders one sentence, transcribes audio, streams a chat reply,
then closes the face. Takes one call slot for a few seconds.
"""
import io
import json
import math
import os
import struct
import sys
import time
import urllib.request
import wave

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7100").rstrip("/")
TOKEN = os.environ["WORKER_TOKEN"]
FACE = os.environ.get("SMOKE_FACE", "/home/khitan/age-progression-app/client/public/examples/3.jpg")


def call(method, path, body=None, ctype=None, timeout=300):
    # Cloudflare's Browser Integrity Check refuses Python's default User-Agent (error 1010).
    req = urllib.request.Request(URL + path, data=body, method=method,
                                 headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "mascotte-smoke/1.0"})
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def tone_wav(sec=3.0, sr=16000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * (0.5 + 0.5 * math.sin(2 * math.pi * 3 * i / sr))
                                                      * math.sin(2 * math.pi * 180 * i / sr))) for i in range(int(sec * sr))))
    return b.getvalue()


ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


s, b, _ = call("GET", "/v1/status")
st = json.loads(b) if s == 200 else {}
check("status", s == 200 and st.get("ready"), json.dumps(st.get("slots")) + f" gpu={st.get('gpu')}")

t = time.time()
s, b, _ = call("POST", "/v1/sessions?steps=4", open(FACE, "rb").read(), "image/jpeg")
sid = json.loads(b).get("id") if s == 200 else None
check("open face", sid, f"{time.time() - t:.1f}s" if sid else b[:200])

if sid:
    t = time.time()
    s, b, h = call("POST", f"/v1/sessions/{sid}/clips?turn=1&seq=0", tone_wav(), "audio/wav")
    check("render one sentence", s == 200 and b[4:8] == b"ftyp",
          f"{len(b) // 1024} KB, {time.time() - t:.1f}s, render {h.get('X-Render-Ms') or h.get('x-render-ms')} ms")
    s, b, _ = call("GET", "/v1/status")
    check("slot counted", json.loads(b)["slots"]["mine"] == 1)

s, b, _ = call("POST", "/v1/transcribe", tone_wav(2.0), "audio/wav")
check("transcribe", s == 200 and "text" in json.loads(b), b[:120])

t = time.time()
s, b, _ = call("POST", "/v1/chat", json.dumps({"messages": [{"role": "user", "content": "Say hello in five words."}],
                                                "options": {"num_predict": 30}}).encode(), "application/json")
text = "".join(json.loads(l).get("message", {}).get("content", "") for l in b.decode().splitlines() if l.strip())
check("chat", s == 200 and text.strip(), f"{time.time() - t:.1f}s: {text.strip()[:60]!r}")

if sid:
    s, _, _ = call("DELETE", f"/v1/sessions/{sid}")
    s2, b, _ = call("GET", "/v1/status")
    check("close face", s == 200 and json.loads(b)["slots"]["mine"] == 0)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
