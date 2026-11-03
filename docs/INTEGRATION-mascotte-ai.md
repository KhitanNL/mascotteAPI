# Adding the video face to mascotte.ai

mascotte.ai keeps its own LLM, ElevenLabs and Gemini Live. The worker only adds
the face: give it a portrait once, then each spoken line as MP3, and it returns
an MP4 of that face saying the line (voice included). This is a separate
"video face" mode next to the WebGL head, not inside it.

Measured on the office RTX 5090: opening a face 1.4 s, a 4 s line rendered in
1.1 s, about 5 calls at once per GPU.

## Settings (Railway)

```
FACE_WORKER_URL=https://face.mascotte.ai
FACE_WORKER_TOKEN=<the worker's token, server side only>
```

## Worker API (all need `Authorization: Bearer $FACE_WORKER_TOKEN`)

| Request | Body | Answer |
|---|---|---|
| `POST /v1/sessions?steps=4` | face image (JPG/PNG, ≤15 MB) | `{id, idle, poster, ...}`: `idle` is a silent MP4 loop of the face at rest (base64), `poster` a JPEG (base64). `503 {"error":"full"}` when the GPU has no free call |
| `POST /v1/sessions/{id}/clips?turn=T&seq=S` | the line's audio: **MP3 as ElevenLabs returns it** (or WAV) | `video/mp4` with the voice; headers `X-Duration-Sec`, `X-Render-Ms` |
| `POST /v1/sessions/{id}/cancel?turn=T` | none | stops rendering an interrupted reply |
| `DELETE /v1/sessions/{id}` | none | ends the call, frees the slot |

### Face settings

`GET /v1/settings` lists them with types, ranges and defaults, so the UI can be
built from it. Pass any of them as query parameters when opening a face; the
answer echoes the values used under `settings`. A bad value is refused with
`400 {"error":"bad_setting","message":...}` before anything touches the GPU.

| Setting | Values | Default | Effect |
|---|---|---|---|
| `quality` | `best`, `fast` | `best` | fast renders about twice as fast with a slightly softer face |
| `variant` | whole number | 42 | a different variant of the head's movement style |
| `autoFrame` | `true`, `false` | `true` | find and frame the face; off only for tight, centred portraits |
| `idleLoop` | 1–6 | 3 | length of the rest loop between replies (× 1.32 s) |
| `resolution` | `512`, `640`, `768`, `1024` | 512 | square video size in px; 512 is the model's native size. GPU time grows with the pixels (640 ≈ 1.6×, 768 ≈ 2.25×, 1024 ≈ 4×, experimental), so larger sizes render slower and fewer calls fit at once |

Example: `POST /v1/sessions?face=3&quality=fast&idleLoop=5&resolution=768`. The
engine's own names (`steps=4|2`, `seed`, `crop=1|0`, `idle_chunks`, `size`)
still work.

Sample faces: `GET /v1/faces` → `{faces:[{id,url}]}`, `GET /v1/faces/{id}` → the
image, `POST /v1/sessions?face={id}` opens one without uploading.

`turn` counts replies in the call (1, 2, 3 …), `seq` counts lines within a reply
(0, 1, 2 …). A clip continues the head's motion from the previous clip of the
same turn; a new turn starts from the rest pose, so it blends with the idle loop.
A face session ends by itself after 10 idle minutes.

## Sidecar routes (server/api.mjs, plain fetch, no new dependency)

```js
const FACE_URL = process.env.FACE_WORKER_URL;
const FACE_TOKEN = process.env.FACE_WORKER_TOKEN;
/** userId -> face session id: a user can only drive their own face. */
const faceSessions = new Map();

function face(path, init = {}) {
  return fetch(FACE_URL + path, {
    ...init,
    headers: { ...init.headers, authorization: `Bearer ${FACE_TOKEN}` },
    signal: AbortSignal.timeout(120_000),
  });
}

// POST /api/face/open   body: the portrait (image/jpeg or image/png)
async function faceOpen(req, res, user, imageBuffer, contentType) {
  const old = faceSessions.get(user.id);
  if (old) face(`/v1/sessions/${old}`, { method: "DELETE" }).catch(() => {});
  const r = await face("/v1/sessions?steps=4", { method: "POST", headers: { "content-type": contentType }, body: imageBuffer });
  if (r.status === 503) return json(res, 503, { error: "busy" }); // all faces in use: try again shortly
  if (!r.ok) return json(res, 502, { error: "face_failed" });
  const s = await r.json();
  faceSessions.set(user.id, s.id);
  json(res, 200, { idle: s.idle, poster: s.poster });
}

// POST /api/face/say?turn=1&seq=0   body: {"text": "...", "voice": "<voice id>"}
async function faceSay(req, res, user, { text, voice }, turn, seq) {
  const id = faceSessions.get(user.id);
  if (!id) return json(res, 409, { error: "no_face" });
  // The same ElevenLabs call /api/tts makes today.
  const tts = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voice)}?output_format=mp3_44100_128`, {
    method: "POST",
    headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY, "content-type": "application/json" },
    body: JSON.stringify({ text, model_id: "eleven_flash_v2_5" }),
  });
  if (!tts.ok) return json(res, 502, { error: "tts_failed" });
  const mp3 = Buffer.from(await tts.arrayBuffer());
  const clip = await face(`/v1/sessions/${id}/clips?turn=${turn}&seq=${seq}`, {
    method: "POST", headers: { "content-type": "audio/mpeg" }, body: mp3,
  });
  if (clip.status === 404) { faceSessions.delete(user.id); return json(res, 409, { error: "no_face" }); } // timed out: reopen
  if (!clip.ok) return json(res, 502, { error: "face_failed" });
  // Metering: seconds of video the user received.
  const seconds = Number(clip.headers.get("x-duration-sec") || 0);
  // meterUsage(user.id, "face_seconds", seconds);
  res.writeHead(200, { "content-type": "video/mp4", "x-duration-sec": String(seconds) });
  res.end(Buffer.from(await clip.arrayBuffer()));
}

// POST /api/face/cancel?turn=1     (user interrupted the reply)
// POST /api/face/close              (left the page, or switched back to the 3D head)
async function faceClose(user) {
  const id = faceSessions.get(user.id);
  faceSessions.delete(user.id);
  if (id) await face(`/v1/sessions/${id}`, { method: "DELETE" }).catch(() => {});
}
```

Notes:
- Send each **sentence** as its own `/api/face/say` as soon as the LLM has it,
  not the whole reply: the face then starts after one sentence (~1.5 s) instead
  of after the whole reply. Fire them in order and play the clips in order.
- The browser plays the clip's own sound; it does not also play the MP3.
- One face per user at a time (the map above). If the sidecar runs more than
  one instance, keep the map in Postgres instead.

## Browser: idle loop and clips

```js
// Two <video> elements stacked: idle (muted loop) underneath, talk on top.
const idle = document.querySelector("#face-idle");   // <video muted loop playsinline>
const talk = document.querySelector("#face-talk");   // <video playsinline>, hidden when not talking
const queue = [];
let playing = false;

function b64ToBlobUrl(b64, type) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type }));
}

async function openFace(portraitBlob) {
  const r = await fetch("/api/face/open", { method: "POST", body: portraitBlob, headers: { "content-type": portraitBlob.type } });
  if (r.status === 503) throw new Error("All faces are busy, try again in a minute");
  const { idle: mp4, poster } = await r.json();
  idle.poster = `data:image/jpeg;base64,${poster}`;
  idle.src = b64ToBlobUrl(mp4, "video/mp4");
  await idle.play();
}

async function say(text, voice, turn, seq) {
  const r = await fetch(`/api/face/say?turn=${turn}&seq=${seq}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, voice }),
  });
  if (!r.ok) throw new Error(`face ${r.status}`);
  queue.push(URL.createObjectURL(await r.blob()));
  if (!playing) playNext();
}

function playNext() {
  const url = queue.shift();
  if (!url) { playing = false; talk.hidden = true; return; }   // back to the idle loop
  playing = true;
  talk.src = url;
  talk.onended = () => { URL.revokeObjectURL(url); playNext(); };
  talk.hidden = false;
  talk.play();
}

function interrupt(turn) {
  queue.splice(0).forEach(URL.revokeObjectURL);
  talk.pause(); talk.hidden = true; playing = false;
  fetch(`/api/face/cancel?turn=${turn}`, { method: "POST" });
}
```

Browsers only allow sound after a click: call `talk.play()` once (it can be on
an empty source) inside the click that starts the conversation.

Lines in one reply: request them in parallel if you like, but **queue the clips
in `seq` order** (keep a small map seq -> url and play when the next one is in).
