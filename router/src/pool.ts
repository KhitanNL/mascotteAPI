/**
 * The pool of GPU workers, as the web app sees it.
 *
 * Each worker (worker/gateway/app.py) reports how many calls it can still
 * take. The pool asks every worker for that every few seconds, sends each new
 * call to the best worker with room, and keeps the call there for its whole
 * length: a call's face lives on one GPU, so every sentence of it must go to
 * the same worker.
 *
 * Best worker: own machines before cloud ones (they cost nothing per hour),
 * then the one with the most free slots. A worker that turns out to be full
 * (its answer can be a few seconds newer than our last look) says so with
 * 503 "full", and the pool simply tries the next.
 *
 * When free slots across the pool run low, onPressure fires: that is where
 * the autoscaler starts a cloud worker, ahead of the moment it is needed.
 */

export type WorkerKind = "own" | "cloud";

export interface WorkerConfig {
  name: string;
  url: string;
  token: string;
  kind: WorkerKind;
}

export interface WorkerState {
  name: string;
  kind: WorkerKind;
  ready: boolean;
  free: number;
  mine: number;
  maxSlots: number;
  /** Calls this pool has open on it. */
  calls: number;
  checkedAt: number | null;
  error?: string;
}

export interface PoolSnapshot {
  workers: WorkerState[];
  freeSlots: number;
  readyWorkers: number;
  openCalls: number;
}

export class NoCapacityError extends Error {
  constructor() {
    super("Every worker is full");
  }
}

export class WorkerError extends Error {
  constructor(readonly worker: string, readonly status: number, message: string) {
    super(`${worker}: ${message}`);
  }
}

export interface PoolOptions {
  fetch?: typeof fetch;
  /** How often to ask every worker for its status. */
  pollMs?: number;
  /** Fire onPressure when free slots across the pool drop to this or below. */
  pressureAt?: number;
  onPressure?: (s: PoolSnapshot) => void;
  /** Any change worth reacting to: a slot freed, a worker came or went. */
  onChange?: (s: PoolSnapshot) => void;
  now?: () => number;
}

/** One live call: its worker and its face session there. */
export interface Call {
  id: string;
  worker: string;
  /** The face at rest (idle loop) and poster, as the worker returned them. */
  opened: Record<string, unknown>;
}

/** Node and the DOM types disagree on Uint8Array generics; at runtime fetch takes it as is. */
const asBody = (b: Uint8Array): BodyInit => b as unknown as BodyInit;

const STATUS_TIMEOUT_MS = 3_000;
const OPEN_TIMEOUT_MS = 60_000;
const CLIP_TIMEOUT_MS = 300_000;

interface Entry {
  cfg: WorkerConfig;
  state: WorkerState;
}

export class WorkerPool {
  private readonly entries = new Map<string, Entry>();
  /** Call id -> worker name. */
  private readonly calls = new Map<string, string>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private readonly fetch: typeof fetch;
  private readonly pollMs: number;
  private readonly pressureAt: number;
  private readonly now: () => number;

  constructor(workers: WorkerConfig[], private readonly opts: PoolOptions = {}) {
    this.fetch = opts.fetch ?? globalThis.fetch.bind(globalThis);
    this.pollMs = opts.pollMs ?? 5_000;
    this.pressureAt = opts.pressureAt ?? 1;
    this.now = opts.now ?? Date.now;
    for (const w of workers) this.add(w);
  }

  // ── Workers ──────────────────────────────────────────────────────────────

  add(cfg: WorkerConfig): void {
    if (this.entries.has(cfg.name)) throw new Error(`worker ${cfg.name} already in the pool`);
    this.entries.set(cfg.name, {
      cfg: { ...cfg, url: cfg.url.replace(/\/+$/, "") },
      state: { name: cfg.name, kind: cfg.kind, ready: false, free: 0, mine: 0, maxSlots: 0, calls: 0, checkedAt: null },
    });
  }

  /** Take a worker out (a cloud machine shutting down). Its calls are forgotten. */
  remove(name: string): void {
    this.entries.delete(name);
    for (const [id, w] of Array.from(this.calls)) if (w === name) this.calls.delete(id);
    this.changed();
  }

  start(): void {
    if (this.timer) return;
    void this.refresh();
    this.timer = setInterval(() => void this.refresh(), this.pollMs);
    (this.timer as { unref?: () => void }).unref?.();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  async refresh(): Promise<PoolSnapshot> {
    await Promise.all(Array.from(this.entries.values(), (e) => this.check(e)));
    return this.changed();
  }

  snapshot(): PoolSnapshot {
    const workers = Array.from(this.entries.values(), (e) => ({ ...e.state }));
    const ready = workers.filter((w) => w.ready);
    return {
      workers,
      freeSlots: ready.reduce((n, w) => n + w.free, 0),
      readyWorkers: ready.length,
      openCalls: this.calls.size,
    };
  }

  /** Which worker holds a call, if any. */
  workerOf(callId: string): string | undefined {
    return this.calls.get(callId);
  }

  // ── Calls ────────────────────────────────────────────────────────────────

  /** Open a call on the best worker with room. Throws NoCapacityError when none has. */
  async openCall(face: Uint8Array, contentType: string, query: Record<string, string> = {}): Promise<Call> {
    const tried = new Set<string>();
    for (;;) {
      const entry = this.candidates().find((e) => !tried.has(e.cfg.name));
      if (!entry) {
        this.changed();
        throw new NoCapacityError();
      }
      tried.add(entry.cfg.name);
      let res: Response;
      try {
        res = await this.request(entry, `/v1/sessions?${new URLSearchParams({ steps: "4", ...query })}`, {
          method: "POST", body: asBody(face), headers: { "content-type": contentType },
        }, OPEN_TIMEOUT_MS);
      } catch (err) {
        this.markDown(entry, err);
        continue;
      }
      if (res.status === 503) {
        // Full (or its face engine is down): our view was stale; try the next one.
        entry.state.free = 0;
        await res.body?.cancel();
        continue;
      }
      if (!res.ok) throw new WorkerError(entry.cfg.name, res.status, await res.text());
      const opened = (await res.json()) as Record<string, unknown>;
      const id = String(opened.id);
      this.calls.set(id, entry.cfg.name);
      entry.state.free = Math.max(0, entry.state.free - 1);
      entry.state.calls += 1;
      this.changed();
      return { id, worker: entry.cfg.name, opened };
    }
  }

  /** One sentence of a reply: WAV in, MP4 of the face saying it out (the worker's response as is). */
  renderClip(callId: string, wav: Uint8Array, turn: number, seq: number, signal?: AbortSignal): Promise<Response> {
    return this.request(this.entryOf(callId), `/v1/sessions/${encodeURIComponent(callId)}/clips?turn=${turn}&seq=${seq}`,
      { method: "POST", body: asBody(wav), headers: { "content-type": "audio/wav" } }, CLIP_TIMEOUT_MS, signal);
  }

  async cancelReply(callId: string, turn: number): Promise<void> {
    const entry = this.entries.get(this.calls.get(callId) ?? "");
    if (!entry) return;
    await this.request(entry, `/v1/sessions/${encodeURIComponent(callId)}/cancel?turn=${turn}`, { method: "POST" },
      STATUS_TIMEOUT_MS).then((r) => r.body?.cancel(), () => {});
  }

  /** End a call and free its slot. Safe to call twice. */
  async closeCall(callId: string): Promise<void> {
    const name = this.calls.get(callId);
    this.calls.delete(callId);
    const entry = name ? this.entries.get(name) : undefined;
    if (!entry) return;
    entry.state.calls = Math.max(0, entry.state.calls - 1);
    entry.state.free = Math.min(entry.state.maxSlots, entry.state.free + 1);
    this.changed();
    await this.request(entry, `/v1/sessions/${encodeURIComponent(callId)}`, { method: "DELETE" }, STATUS_TIMEOUT_MS)
      .then((r) => r.body?.cancel(), () => {});
  }

  /** Speech to text, on the call's own worker when there is one (its GPU already serves this caller). */
  transcribe(audio: Uint8Array, contentType: string, callId?: string): Promise<Response> {
    return this.request(this.anyEntry(callId), "/v1/transcribe",
      { method: "POST", body: asBody(audio), headers: { "content-type": contentType } }, 30_000);
  }

  /** The language model (Ollama /api/chat body), streamed back as NDJSON. */
  chat(body: Record<string, unknown>, callId?: string, signal?: AbortSignal): Promise<Response> {
    return this.request(this.anyEntry(callId), "/v1/chat",
      { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } }, 120_000, signal);
  }

  // ── Internals ────────────────────────────────────────────────────────────

  private candidates(): Entry[] {
    return Array.from(this.entries.values())
      .filter((e) => e.state.ready && e.state.free > 0)
      .sort((a, b) => (a.cfg.kind === b.cfg.kind ? b.state.free - a.state.free : a.cfg.kind === "own" ? -1 : 1));
  }

  private entryOf(callId: string): Entry {
    const entry = this.entries.get(this.calls.get(callId) ?? "");
    if (!entry) throw new WorkerError("pool", 404, `no such call ${callId}`);
    return entry;
  }

  private anyEntry(callId?: string): Entry {
    const own = callId ? this.entries.get(this.calls.get(callId) ?? "") : undefined;
    if (own) return own;
    const ready = Array.from(this.entries.values()).filter((e) => e.state.ready);
    if (!ready.length) throw new NoCapacityError();
    // Own machines first, then the least busy.
    ready.sort((a, b) => (a.cfg.kind === b.cfg.kind ? a.state.calls - b.state.calls : a.cfg.kind === "own" ? -1 : 1));
    return ready[0];
  }

  private async check(e: Entry): Promise<void> {
    try {
      const res = await this.request(e, "/v1/status", { method: "GET" }, STATUS_TIMEOUT_MS);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const s = (await res.json()) as { ready?: boolean; slots?: { free?: number; mine?: number; maxSlots?: number } };
      e.state.ready = s.ready === true;
      e.state.free = s.slots?.free ?? 0;
      e.state.mine = s.slots?.mine ?? 0;
      e.state.maxSlots = s.slots?.maxSlots ?? 0;
      e.state.checkedAt = this.now();
      e.state.error = s.ready ? undefined : "engines not ready";
    } catch (err) {
      this.markDown(e, err);
    }
  }

  private markDown(e: Entry, err: unknown): void {
    e.state.ready = false;
    e.state.free = 0;
    e.state.checkedAt = this.now();
    e.state.error = err instanceof Error ? err.message : String(err);
  }

  private request(e: Entry, path: string, init: RequestInit, timeoutMs: number, signal?: AbortSignal): Promise<Response> {
    const timeout = AbortSignal.timeout(timeoutMs);
    return this.fetch(e.cfg.url + path, {
      ...init,
      headers: { ...(init.headers as Record<string, string> | undefined), authorization: `Bearer ${e.cfg.token}` },
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  }

  private changed(): PoolSnapshot {
    const s = this.snapshot();
    this.opts.onChange?.(s);
    if (s.freeSlots <= this.pressureAt) this.opts.onPressure?.(s);
    return s;
  }
}
