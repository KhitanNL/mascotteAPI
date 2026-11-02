import { afterEach, describe, expect, it, vi } from "vitest";
import { NoCapacityError, WaitingRoom, WaitTimeoutError, WorkerPool, workersFromEnv, type WorkerConfig } from "../src";
import { FACE, FakeWorker, fakeFetch } from "./fakeWorkers";

const cfg = (name: string, kind: "own" | "cloud" = "own"): WorkerConfig => ({ name, kind, url: `http://${name}`, token: "tok" });

async function poolOf(workers: FakeWorker[], kinds: Record<string, "own" | "cloud"> = {}, opts = {}) {
  const pool = new WorkerPool(workers.map((w) => cfg(w.name, kinds[w.name] ?? "own")), { fetch: fakeFetch(workers), ...opts });
  await pool.refresh();
  return pool;
}

describe("WorkerPool", () => {
  it("prefers own machines, then cloud, and spills over when one is full", async () => {
    const office = new FakeWorker("office", 2);
    const cloud = new FakeWorker("cloud1", 5);
    const pool = await poolOf([cloud, office], { cloud1: "cloud" });
    const a = await pool.openCall(FACE, "image/jpeg");
    const b = await pool.openCall(FACE, "image/jpeg");
    const c = await pool.openCall(FACE, "image/jpeg");
    expect([a.worker, b.worker, c.worker]).toEqual(["office", "office", "cloud1"]);
    expect(pool.snapshot().openCalls).toBe(3);
  });

  it("throws NoCapacityError when every worker is full", async () => {
    const pool = await poolOf([new FakeWorker("office", 1)]);
    await pool.openCall(FACE, "image/jpeg");
    await expect(pool.openCall(FACE, "image/jpeg")).rejects.toBeInstanceOf(NoCapacityError);
  });

  it("tries the next worker when one turns out to be full despite its last status", async () => {
    const office = new FakeWorker("office", 1);
    const cloud = new FakeWorker("cloud1", 1);
    const pool = await poolOf([office, cloud], { cloud1: "cloud" });
    office.sessions.add("someone-else"); // filled up since we last looked
    const call = await pool.openCall(FACE, "image/jpeg");
    expect(call.worker).toBe("cloud1");
  });

  it("skips a worker that is down and marks it so", async () => {
    const office = new FakeWorker("office", 2);
    const cloud = new FakeWorker("cloud1", 2);
    const pool = await poolOf([office, cloud], { cloud1: "cloud" });
    office.down = true;
    expect((await pool.openCall(FACE, "image/jpeg")).worker).toBe("cloud1");
    expect(pool.snapshot().workers.find((w) => w.name === "office")?.ready).toBe(false);
  });

  it("keeps every sentence of a call on its own worker", async () => {
    const office = new FakeWorker("office", 1);
    const cloud = new FakeWorker("cloud1", 1);
    const pool = await poolOf([office, cloud], { cloud1: "cloud" });
    await pool.openCall(FACE, "image/jpeg");
    const second = await pool.openCall(FACE, "image/jpeg");
    const res = await pool.renderClip(second.id, new Uint8Array([1]), 1, 0);
    expect(res.headers.get("content-type")).toBe("video/mp4");
    expect(cloud.requests).toContain("POST /v1/sessions/cloud1-s1/clips");
    expect(office.requests.some((r) => r.includes("/clips"))).toBe(false);
  });

  it("frees the slot when a call closes, and closing twice is harmless", async () => {
    const office = new FakeWorker("office", 1);
    const pool = await poolOf([office]);
    const call = await pool.openCall(FACE, "image/jpeg");
    await pool.closeCall(call.id);
    await pool.closeCall(call.id);
    expect(office.sessions.size).toBe(0);
    expect((await pool.openCall(FACE, "image/jpeg")).worker).toBe("office");
  });

  it("sends speech and chat to the call's worker, or the least busy own one", async () => {
    const office = new FakeWorker("office", 1);
    const cloud = new FakeWorker("cloud1", 2);
    const pool = await poolOf([office, cloud], { cloud1: "cloud" });
    await pool.openCall(FACE, "image/jpeg");
    const onCloud = await pool.openCall(FACE, "image/jpeg");
    expect((await (await pool.transcribe(new Uint8Array([1]), "audio/webm", onCloud.id)).json()).text).toBe("heard by cloud1");
    expect(await (await pool.chat({ messages: [] })).text()).toContain("from office");
  });

  it("fires onPressure when free slots run low", async () => {
    const onPressure = vi.fn();
    const pool = await poolOf([new FakeWorker("office", 3)], {}, { pressureAt: 1, onPressure });
    onPressure.mockClear();
    await pool.openCall(FACE, "image/jpeg");
    expect(onPressure).not.toHaveBeenCalled();
    await pool.openCall(FACE, "image/jpeg");
    expect(onPressure).toHaveBeenCalledWith(expect.objectContaining({ freeSlots: 1 }));
  });

  it("adds and removes workers at runtime (cloud machines coming and going)", async () => {
    const office = new FakeWorker("office", 1);
    const cloud = new FakeWorker("cloud1", 2);
    const pool = new WorkerPool([cfg("office")], { fetch: fakeFetch([office, cloud]) });
    await pool.refresh();
    await pool.openCall(FACE, "image/jpeg");
    await expect(pool.openCall(FACE, "image/jpeg")).rejects.toBeInstanceOf(NoCapacityError);
    pool.add(cfg("cloud1", "cloud"));
    await pool.refresh();
    const call = await pool.openCall(FACE, "image/jpeg");
    expect(call.worker).toBe("cloud1");
    pool.remove("cloud1");
    expect(pool.workerOf(call.id)).toBeUndefined();
  });
});

describe("WaitingRoom", () => {
  afterEach(() => vi.useRealTimers());

  it("gives a call at once when there is room", async () => {
    const pool = await poolOf([new FakeWorker("office", 1)]);
    const room = new WaitingRoom(pool);
    expect((await room.enter(FACE, "image/jpeg", { tier: "free" })).worker).toBe("office");
    room.stop();
  });

  it("queues when full, paid users first, and hands over freed slots in that order", async () => {
    const office = new FakeWorker("office", 1);
    const pool = await poolOf([office]);
    const room = new WaitingRoom(pool, { pollMs: 60_000 });
    const first = await room.enter(FACE, "image/jpeg", { tier: "free" });

    const order: string[] = [];
    const positions: Record<string, number[]> = { free: [], paid: [] };
    const free = room.enter(FACE, "image/jpeg", { tier: "free", onPosition: (p) => positions.free.push(p) })
      .then((c) => (order.push("free"), c));
    const paid = room.enter(FACE, "image/jpeg", { tier: "paid", onPosition: (p) => positions.paid.push(p) })
      .then((c) => (order.push("paid"), c));
    await new Promise((r) => setTimeout(r, 10));
    expect(room.waiting()).toEqual({ paid: 1, free: 1 });
    expect(positions.paid.at(-1)).toBe(1);
    expect(positions.free.at(-1)).toBe(2);

    await pool.closeCall(first.id);
    room.nudge();
    const paidCall = await paid;
    await pool.closeCall(paidCall.id);
    room.nudge();
    await free;
    expect(order).toEqual(["paid", "free"]);
    room.stop();
  });

  it("lets a caller leave the line", async () => {
    const pool = await poolOf([new FakeWorker("office", 1)]);
    const room = new WaitingRoom(pool, { pollMs: 60_000 });
    await room.enter(FACE, "image/jpeg", { tier: "paid" });
    const ctrl = new AbortController();
    const waiting = room.enter(FACE, "image/jpeg", { tier: "paid", signal: ctrl.signal });
    ctrl.abort(new Error("closed the page"));
    await expect(waiting).rejects.toThrow("closed the page");
    expect(room.length).toBe(0);
    room.stop();
  });

  it("gives up after maxWaitMs", async () => {
    const pool = await poolOf([new FakeWorker("office", 1)]);
    const room = new WaitingRoom(pool, { maxWaitMs: 50, pollMs: 60_000 });
    await room.enter(FACE, "image/jpeg", { tier: "paid" });
    await expect(room.enter(FACE, "image/jpeg", { tier: "paid" })).rejects.toBeInstanceOf(WaitTimeoutError);
    room.stop();
  });

  it("picks up a slot a worker freed on its own, by polling", async () => {
    const office = new FakeWorker("office", 1);
    const pool = await poolOf([office]);
    const room = new WaitingRoom(pool, { pollMs: 20 });
    const first = await room.enter(FACE, "image/jpeg", { tier: "paid" });
    const waiting = room.enter(FACE, "image/jpeg", { tier: "paid" });
    office.sessions.delete(first.id); // the worker timed the idle call out
    expect((await waiting).worker).toBe("office");
    room.stop();
  });
});

describe("workersFromEnv", () => {
  it("reads name|kind|url|token entries", () => {
    expect(workersFromEnv("office|own|http://127.0.0.1:7100|abc; c1|cloud|https://x|def")).toEqual([
      { name: "office", kind: "own", url: "http://127.0.0.1:7100", token: "abc" },
      { name: "c1", kind: "cloud", url: "https://x", token: "def" },
    ]);
    expect(() => workersFromEnv("office|gpu|http://x|t")).toThrow();
  });
});
