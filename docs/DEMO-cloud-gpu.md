# Demo day: extra cloud GPU behind face.mascotte.ai

The demo keeps using https://face.mascotte.ai. During the demo the office
gateway sends marked calls to a RunPod RTX 5090 first and to the office GPU
when the cloud is full or down. Only calls opened with `cloud=1` are marked:
mascotte.ai adds it for the accounts in `FACE_CLOUD_EMAILS` (only
khitans@gmail.com), so nobody else uses the paid GPU.

mascotte.ai sidecar, in `faceOpen`:

```js
const cloudUsers = (process.env.FACE_CLOUD_EMAILS || "").toLowerCase().split(",").map((s) => s.trim()).filter(Boolean);
const cloud = cloudUsers.includes(String(user.email || "").toLowerCase()) ? "&cloud=1" : "";
const r = await face(`/v1/sessions?steps=4${cloud}`, { method: "POST", headers: { "content-type": contentType }, body: imageBuffer });
```

## Once: image and template

- Image: `ghcr.io/khitannl/mascotte-worker:demo` (private, built from `docker/Dockerfile`, models baked in).
- RunPod template: that image, registry credential for ghcr.io, HTTP port 7100,
  container disk 60 GB, env `WORKER_TOKEN` (a RunPod secret, its own token, not the office one).
- Pod: Secure Cloud, on demand, EU, RTX 5090. About $1.20 per hour. Keep at least $20 balance (pods stop at $0).

## Demo day

| When | Step | Check |
|---|---|---|
| 15:30 | Start the pod from the template | RunPod shows it running |
| 15:30 to ~15:40 | Wait until it is ready | `curl -s https://<pod-id>-7100.proxy.runpod.net/ping` answers `{"ok":true}` |
| ~15:40 | Put it behind the office: `scripts/cloud.sh add demo https://<pod-id>-7100.proxy.runpod.net` (paste its token when asked) | `scripts/cloud.sh status` shows `demo` with `ready: True` |
| 15:50 | Test call from a laptop and a phone on the demo | `scripts/cloud.sh status` shows the cloud with fewer free calls during the call |
| 16:00 to 20:00 | Demo | If the cloud fails, new calls go to the office by themselves |
| 20:00 | `scripts/cloud.sh off demo` | No new calls go to the cloud |
| when its calls end | Stop (terminate) the pod in RunPod | Billing stops; note the cost |

## If something goes wrong

- Cloud slow or broken: `scripts/cloud.sh off demo`. New calls go to the office.
  The office has 2 calls for mascotte while the S4YFS study runs (until 29 Nov, maybe 6 Dec).
- Office worker restarts: only when `curl -s http://127.0.0.1:7012/health` shows `"sessions":0`.
- Never paste tokens in chat or logs.

## Measured (fill in after the test session)

- Pod start to ready: _
- Calls per cloud RTX 5090 without stutter: _
- Cost of the test session: _
