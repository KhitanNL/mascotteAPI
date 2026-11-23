#!/usr/bin/env python3
"""The demo's extra cloud GPU on RunPod, from start to stop. See docs/DEMO-cloud-gpu.md.

  python3 scripts/runpod.py gpus                 what is free now, with prices
  python3 scripts/runpod.py create [--max 0.90]  rent the first free GPU from PREFER under the cap
  python3 scripts/runpod.py create --gpu "NVIDIA RTX A6000" --cloud SECURE   a specific one
  python3 scripts/runpod.py status               pod state, rate, and whether the worker answers
  python3 scripts/runpod.py connect              list it on face.mascotte.ai as cloud worker "demo"
  python3 scripts/runpod.py disconnect           no new calls on it (open calls finish)
  python3 scripts/runpod.py stop | start         stop keeps the pod (GPU may be gone on start)
  python3 scripts/runpod.py delete               remove the pod for good
  python3 scripts/runpod.py cost [--hours 24]    what RunPod billed

Secrets: .runpod-api-key (RunPod key, scopes pods:write, catalog:read, billing:read) and
.runpod-token (the worker token the pod's gateway checks). Both chmod 600, gitignored, never printed.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.runpod.io/v2"
POD_FILE = os.path.join(ROOT, ".runpod-pod")
CLOUD_FILE = os.environ.get("CLOUD_WORKERS_FILE", os.path.join(ROOT, ".cloud-workers"))
NAME = "demo"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
BOOT = ('bash -c "mkdir -p /workspace; (curl -fsSL https://raw.githubusercontent.com/KhitanNL/'
        'mascotteAPI/main/docker/pod-bootstrap.sh | bash > /workspace/boot.log 2>&1 &); /start.sh"')
# Best first. FlashHead Lite needs a recent NVIDIA card; 5090 is what the office runs.
PREFER = [
    ("NVIDIA GeForce RTX 5090", "SECURE"),
    ("NVIDIA GeForce RTX 5090", "COMMUNITY"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ("NVIDIA L40S", "SECURE"),
    ("NVIDIA RTX 6000 Ada Generation", "SECURE"),
    ("NVIDIA RTX PRO 4500 Blackwell Server Edition", "SECURE"),
    ("NVIDIA RTX A6000", "SECURE"),
]
EU_FIRST = ("EU", "EUR")


def secret(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read().strip()


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"authorization": "Bearer " + secret(".runpod-api-key"),
                                          "content-type": "application/json",
                                          "user-agent": "mascotte-ops/1"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        for s in (".runpod-token", ".runpod-api-key"):
            try:
                text = text.replace(secret(s), "<secret>")
            except OSError:
                pass
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, {"detail": text[:500]}


def pod_id():
    if not os.path.exists(POD_FILE):
        sys.exit("No pod recorded (.runpod-pod). Run create first.")
    return open(POD_FILE).read().strip()


def proxy_url(pid):
    return f"https://{pid}-7100.proxy.runpod.net"


def catalog(cloud):
    st, d = call("GET", f"/catalog/gpus?include=AVAILABILITY&product=POD&cloud={cloud}")
    if st != 200:
        sys.exit(f"catalog {cloud}: HTTP {st} {d.get('detail')}")
    return {g["id"]: g for g in d.get("gpus", [])}


def cmd_gpus(a):
    for cloud in ("SECURE", "COMMUNITY"):
        cat = catalog(cloud)
        print(f"== {cloud}")
        for gid, c in PREFER:
            if c != cloud or gid not in cat:
                continue
            g = cat[gid]
            dcs = [x["id"] for x in g.get("dataCenters", []) if x.get("availability") != "NONE"]
            print(f"  {gid:<46} ${g['price'][cloud.lower()]}/h  {g.get('availability')}  {' '.join(dcs[:6])}")


def cmd_create(a):
    if os.path.exists(POD_FILE):
        st, p = call("GET", f"/pods/{pod_id()}")
        if st == 200 and p.get("status") != "TERMINATED":
            sys.exit(f"Pod {p['id']} already exists ({p['status']}). Use start, or delete it first.")
    choices = [(a.gpu, a.cloud)] if a.gpu else PREFER
    cats = {c: catalog(c) for c in ("SECURE", "COMMUNITY")}
    for gid, cloud in choices:
        g = cats[cloud].get(gid)
        if not g or g.get("availability") in (None, "NONE"):
            continue
        price = g["price"][cloud.lower()]
        if price > a.max:
            print(f"skip {gid} {cloud}: ${price}/h is over the ${a.max} cap")
            continue
        dcs = [x["id"] for x in g.get("dataCenters", []) if x.get("availability") != "NONE"]
        eu = [d for d in dcs if d.startswith(EU_FIRST)]
        for attempt in ([eu] if eu else []) + [None]:
            body = {"name": "mascotte-demo", "image": IMAGE, "cloud": cloud,
                    "gpu": {"id": gid, "count": 1}, "disk": 60, "ports": ["7100/http"],
                    "env": {"WORKER_TOKEN": secret(".runpod-token")}, "args": BOOT}
            if attempt:
                body["dataCenterIds"] = attempt
            st, p = call("POST", "/pods", body)
            if st in (200, 201, 202) and p.get("id"):
                if p.get("cost") and p["cost"] > a.max:
                    call("DELETE", f"/pods/{p['id']}")
                    sys.exit(f"RunPod billed ${p['cost']}/h, over the cap: deleted it again.")
                with open(POD_FILE, "w") as f:
                    f.write(p["id"])
                print(f"created {p['id']}: {gid} {cloud} {p.get('dataCenterId') or ''} ${p.get('cost')}/h")
                print(f"worker URL {proxy_url(p['id'])} (ready in ~10 min; run status)")
                return
            print(f"  {gid} {cloud} {'EU' if attempt else 'any region'}: HTTP {st} {p.get('detail')}")
    sys.exit("Nothing free under the cap. Run gpus, then create --gpu ... --max ...")


def ping(pid):
    try:
        req = urllib.request.Request(proxy_url(pid) + "/v1/status", headers={
            "authorization": "Bearer " + secret(".runpod-token"), "user-agent": "mascotte-ops/1"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001 (any failure means not ready)
        return {"error": type(e).__name__}


def cmd_status(a):
    pid = pod_id()
    st, p = call("GET", f"/pods/{pid}")
    if st != 200:
        sys.exit(f"HTTP {st} {p.get('detail')}")
    print(f"pod {pid}: {p.get('status')}  {p.get('gpu', {}).get('id')}  {p.get('cloud')} "
          f"{p.get('dataCenterId') or ''}  ${p.get('cost')}/h  started {p.get('startedAt')}")
    s = ping(pid)
    if "error" in s:
        print(f"worker: not answering yet ({s['error']}); first start takes ~10 min")
    else:
        print(f"worker: ready={s.get('ready')} slots={s.get('slots')} gpu={s.get('gpu')}")
    print(f"listed on face.mascotte.ai: {listed()}")


def read_lines():
    return open(CLOUD_FILE).read().splitlines() if os.path.exists(CLOUD_FILE) else []


def write_lines(lines):
    fd = os.open(CLOUD_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    os.chmod(CLOUD_FILE, 0o600)


def listed():
    for ln in read_lines():
        if ln.lstrip("#").split("|")[0].strip() == NAME:
            return "off (open calls finish)" if ln.startswith("#") else "on"
    return "no"


def cmd_connect(a):
    pid = pod_id()
    s = ping(pid)
    if "error" in s and not a.force:
        sys.exit("The pod's worker does not answer yet; wait for status to say ready (or --force).")
    lines = [ln for ln in read_lines() if ln.lstrip("#").split("|")[0].strip() != NAME]
    lines.append(f"{NAME}|{proxy_url(pid)}|{secret('.runpod-token')}")
    write_lines(lines)
    print(f"cloud worker '{NAME}' on: {proxy_url(pid)}. Only calls opened with cloud=1 use it.")


def cmd_disconnect(a):
    lines = [("#" + ln if ln.split("|")[0].strip() == NAME else ln) for ln in read_lines()]
    write_lines(lines)
    print(f"cloud worker '{NAME}': {listed()}")


def action(name):
    pid = pod_id()
    st, p = call("POST", f"/pods/{pid}/action", {"action": name})
    print(f"{name}: HTTP {st} {p.get('status') or p.get('detail')}")


def cmd_delete(a):
    pid = pod_id()
    st, p = call("DELETE", f"/pods/{pid}")
    print(f"delete {pid}: HTTP {st} {'' if st == 204 else p.get('detail')}")
    if st in (204, 404):
        os.remove(POD_FILE)


def cmd_cost(a):
    end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - a.hours * 3600))
    st, d = call("GET", f"/billing?startTime={start}&endTime={end}&bucketSize=hour")
    if st != 200:
        sys.exit(f"HTTP {st} {d.get('detail')}")
    tot = sum(float(r.get("totalAmount") or 0) for r in d.get("records", []))
    print(f"RunPod billed ${tot:.2f} in the last {a.hours} h (records may lag ~1 h)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    c = sub.add_parser("create")
    c.add_argument("--max", type=float, default=0.90, help="highest $/h to accept")
    c.add_argument("--gpu")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    sub.add_parser("status")
    cn = sub.add_parser("connect")
    cn.add_argument("--force", action="store_true")
    for n in ("disconnect", "stop", "start", "delete"):
        sub.add_parser(n)
    co = sub.add_parser("cost")
    co.add_argument("--hours", type=int, default=24)
    a = ap.parse_args()
    {"gpus": cmd_gpus, "create": cmd_create, "status": cmd_status, "connect": cmd_connect,
     "disconnect": cmd_disconnect, "stop": lambda a: action("stop"), "start": lambda a: action("start"),
     "delete": cmd_delete, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    main()
