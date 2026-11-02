/**
 * The waiting room: when every worker is full, callers queue here instead of
 * being turned away, and get a call as soon as a slot frees or a cloud worker
 * comes up. Paying users go ahead of free ones; within each, first come first
 * served.
 *
 * The pool tells us when something changed (a call ended, a worker reported
 * free slots, a worker joined); we also look again every pollMs in case a
 * worker freed a slot on its own (an idle call it timed out).
 */
import { NoCapacityError, type Call, type WorkerPool } from "./pool";

export type Tier = "paid" | "free";

export interface WaitOptions {
  tier: Tier;
  /** Abort to leave the queue (the caller closed the page). */
  signal?: AbortSignal;
  /** Called with the caller's place in line (1 = next) whenever it changes. */
  onPosition?: (position: number) => void;
  query?: Record<string, string>;
}

export class WaitTimeoutError extends Error {
  constructor() {
    super("No slot became free in time");
  }
}

interface Waiter {
  tier: Tier;
  seq: number;
  face: Uint8Array;
  contentType: string;
  opts: WaitOptions;
  resolve: (c: Call) => void;
  reject: (e: unknown) => void;
  timer: ReturnType<typeof setTimeout>;
  lastPosition: number;
}

export class WaitingRoom {
  private queue: Waiter[] = [];
  private seq = 0;
  private pumping = false;
  private again = false;
  private poller: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly pool: WorkerPool,
    private readonly opts: { maxWaitMs?: number; pollMs?: number } = {},
  ) {}

  get length(): number {
    return this.queue.length;
  }

  /** How many are waiting, per tier: the autoscaler's other signal. */
  waiting(): { paid: number; free: number } {
    return {
      paid: this.queue.filter((w) => w.tier === "paid").length,
      free: this.queue.filter((w) => w.tier === "free").length,
    };
  }

  /** Get a call: at once when there is room, otherwise after waiting in line. */
  enter(face: Uint8Array, contentType: string, opts: WaitOptions): Promise<Call> {
    return new Promise<Call>((resolve, reject) => {
      if (opts.signal?.aborted) return reject(opts.signal.reason);
      const w: Waiter = {
        tier: opts.tier, seq: ++this.seq, face, contentType, opts, resolve, reject, lastPosition: 0,
        timer: setTimeout(() => this.leave(w, new WaitTimeoutError()), this.opts.maxWaitMs ?? 10 * 60_000),
      };
      opts.signal?.addEventListener("abort", () => this.leave(w, opts.signal!.reason), { once: true });
      this.queue.push(w);
      this.queue.sort((a, b) => (a.tier === b.tier ? a.seq - b.seq : a.tier === "paid" ? -1 : 1));
      this.ensurePolling();
      void this.pump();
    });
  }

  /** Something changed in the pool: hand free slots to the people in line. Wire to pool onChange. */
  nudge(): void {
    if (this.queue.length) void this.pump();
  }

  stop(): void {
    if (this.poller) clearInterval(this.poller);
    this.poller = null;
  }

  private async pump(): Promise<void> {
    if (this.pumping) {
      this.again = true;
      return;
    }
    this.pumping = true;
    try {
      do {
        this.again = false;
        while (this.queue.length) {
          const w = this.queue[0];
          try {
            const call = await this.pool.openCall(w.face, w.contentType, w.opts.query);
            if (this.queue[0] === w) {
              this.queue.shift();
              clearTimeout(w.timer);
              w.resolve(call);
            } else {
              // They left while we were opening it: give the slot back.
              await this.pool.closeCall(call.id);
            }
          } catch (err) {
            if (err instanceof NoCapacityError) break;
            if (this.queue[0] === w) {
              this.queue.shift();
              clearTimeout(w.timer);
              w.reject(err);
            }
          }
        }
      } while (this.again);
    } finally {
      this.pumping = false;
      this.tellPositions();
      if (!this.queue.length) this.stop();
    }
  }

  private leave(w: Waiter, reason: unknown): void {
    const i = this.queue.indexOf(w);
    if (i < 0) return;
    this.queue.splice(i, 1);
    clearTimeout(w.timer);
    w.reject(reason);
    this.tellPositions();
  }

  private tellPositions(): void {
    this.queue.forEach((w, i) => {
      if (w.lastPosition !== i + 1) {
        w.lastPosition = i + 1;
        w.opts.onPosition?.(i + 1);
      }
    });
  }

  private ensurePolling(): void {
    if (this.poller) return;
    this.poller = setInterval(() => {
      void this.pool.refresh().then(() => this.nudge());
    }, this.opts.pollMs ?? 3_000);
    (this.poller as { unref?: () => void }).unref?.();
  }
}
