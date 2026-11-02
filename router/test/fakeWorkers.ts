/** Workers in memory, speaking the gateway's API, for the router tests. */
export class FakeWorker {
  sessions = new Set<string>();
  ready = true;
  down = false;
  /** Report this many free slots regardless of the truth (a stale or lying status). */
  claimFree: number | null = null;
  private n = 0;
  requests: string[] = [];

  constructor(readonly name: string, readonly maxSlots: number) {}

  get free(): number {
    return Math.max(0, this.maxSlots - this.sessions.size);
  }

  handle(method: string, path: string): Response {
    this.requests.push(`${method} ${path.split("?")[0]}`);
    const p = path.split("?")[0];
    if (p === "/v1/status") {
      return Response.json({ ready: this.ready, slots: { free: this.claimFree ?? this.free, mine: this.sessions.size, maxSlots: this.maxSlots } });
    }
    if (p === "/v1/sessions" && method === "POST") {
      if (this.free <= 0) return Response.json({ error: "full" }, { status: 503 });
      const id = `${this.name}-s${++this.n}`;
      this.sessions.add(id);
      return Response.json({ id, idle: "AAAA", poster: "BBBB" });
    }
    const m = p.match(/^\/v1\/sessions\/([^/]+)(\/clips|\/cancel)?$/);
    if (m) {
      const id = decodeURIComponent(m[1]);
      if (method === "DELETE") {
        this.sessions.delete(id);
        return Response.json({ ok: true });
      }
      if (m[2] === "/clips") {
        return this.sessions.has(id)
          ? new Response(new Uint8Array([0, 0, 0, 0, 0x66, 0x74, 0x79, 0x70]), { headers: { "content-type": "video/mp4" } })
          : Response.json({ detail: "no such session" }, { status: 404 });
      }
      return Response.json({ ok: true });
    }
    if (p === "/v1/transcribe") return Response.json({ text: `heard by ${this.name}` });
    if (p === "/v1/chat") return new Response(`{"message":{"content":"from ${this.name}"}}\n{"done":true}\n`);
    return new Response("not found", { status: 404 });
  }
}

/** A fetch that routes http://<worker name>/... to the fake worker of that name. */
export function fakeFetch(workers: FakeWorker[], token = "tok"): typeof fetch {
  const byHost = new Map(workers.map((w) => [w.name, w]));
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const w = byHost.get(url.host);
    if (!w || w.down) throw new TypeError("fetch failed");
    const headers = new Headers(init?.headers);
    if (headers.get("authorization") !== `Bearer ${token}`) return new Response("bad token", { status: 401 });
    return w.handle(init?.method ?? "GET", url.pathname + url.search);
  }) as typeof fetch;
}

export const FACE = new Uint8Array([0xff, 0xd8, 0xff]);
