# Demo day: extra cloud GPU behind face.mascotte.ai

Plan (agreed 23 Nov 2026): start at **12:00 on 24 Nov**, working by 14:00. Demo 16:00 to 20:00.

The demo keeps using https://face.mascotte.ai. The office gateway sends marked calls to the
RunPod GPU first and to the office GPU when the cloud is full or down. Only calls opened with
`cloud=1` are marked: mascotte.ai adds it for the accounts in `FACE_CLOUD_EMAILS` (only the
owner's account), so nobody else uses the paid GPU.

## Before 12:00: ElevenLabs credit

mascotte.ai's voice is ElevenLabs. With no credit the avatar opens but never speaks (no greeting);
the video side looks fine. Top it up for 16:00 to 20:00 of talking. (Happened 23 Nov.)

## Before 12:00: is mascotte.ai sending cloud=1?

**Not built yet (23 Nov).** Without it no call ever uses the cloud GPU. The request for the
mascotte.ai side (sidecar `faceOpen` + Railway env `FACE_CLOUD_EMAILS`, plus the cloud row on the
"Video GPUs" panel) is in the shared notes file `mascotte-ai-cloud-gpu-request.md`. Snippet:

```js
const cloudUsers = (process.env.FACE_CLOUD_EMAILS || "").toLowerCase().split(",").map((s) => s.trim()).filter(Boolean);
const cloud = cloudUsers.includes(String(user.email || "").toLowerCase()) ? "&cloud=1" : "";
const r = await face(`/v1/sessions?steps=4${cloud}`, { method: "POST", headers: { "content-type": contentType }, body: imageBuffer });
```

## How it runs

- No image, no template: a stock `runpod/pytorch` pod runs `docker/pod-bootstrap.sh` from GitHub,
  which installs the face engine (FlashHead Lite) and the gateway and serves the worker API on :7100.
  First start ~10 min.
- Everything is done with `scripts/runpod.py` from WSL in `/home/khitan/mascotteAPI`
  (RunPod API v2; v1 was retired 15 Nov 2026).
- Secrets, chmod 600, gitignored, never printed: `.runpod-api-key` (RunPod key "mascotte-demo",
  scopes pods, catalog read, billing read; re-save with `scripts/runpod-key.sh`), `.runpod-token`
  (worker token for the pod), `.runpod-pod` (pod id), `.cloud-workers` (gateway list).
- Approved budget: about $0.90 per hour (`create --max 0.90` is the default cap). Ask before going higher.

## 24 Nov

| When | Step | Check |
|---|---|---|
| 12:00 | `python3 scripts/runpod.py gpus` | What is free and at what price |
| 12:00 | `python3 scripts/runpod.py create` (tries 5090, then 4090, L40S, ... under the cap, EU first) | Prints pod id and $/h |
| ~12:10 | `python3 scripts/runpod.py status` until `worker: ready=True` | If not after 20 min: RunPod web terminal, `tail -50 /workspace/boot.log` |
| ~12:15 | `python3 scripts/runpod.py connect` | `status` says `listed on face.mascotte.ai: on`; office `/v1/status` shows `cloud` with `ready: true` |
| 12:20 to 14:00 | Test calls from the owner's account on mascotte.ai (laptop + phone), then several at once | Office `/v1/status` cloud `held` goes up; no stutter; note calls per GPU |
| 16:00 to 20:00 | Demo | If the cloud fails, new calls go to the office by themselves |
| 20:00 | `python3 scripts/runpod.py disconnect` | No new calls on the cloud |
| when its calls end | `python3 scripts/runpod.py delete`, then `cost --hours 12` | Report the cost |

Price and stock change by the hour (23 Nov 16:30: Secure 5090 $1.39/h; Community 5090 $0.69/h;
Secure 4090 $0.89/h in EU-CZ-1). Stopping a pod and starting it later can fail when the GPU is
taken, so on demo day keep the pod running from 12:00 until the demo ends.

## If something goes wrong

- Cloud slow or broken: `python3 scripts/runpod.py disconnect`. New calls go to the office.
  The office has 2 calls for mascotte while the S4YFS study runs (until 29 Nov, maybe 6 Dec).
- Office worker restarts: only when `curl -s http://127.0.0.1:7012/health` shows `"sessions":0`.
- Never paste tokens in chat or logs.

## Measured (fill in on 24 Nov)

- GPU, cloud, data center, $/h: _
- Pod create to ready: _
- Calls per cloud GPU without stutter: _
- Total cost: _
