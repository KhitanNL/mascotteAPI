/**
 * The router against real workers: npm run demo   (reads MASCOTTE_WORKERS, or the local worker from ../.env)
 *
 * Opens N calls at once through the waiting room (default 4, more than the
 * office worker's 3 slots, so the last one waits), renders one sentence on
 * each, and closes them one by one, printing which worker served each call
 * and how long each waited.
 */
import { readFileSync } from "node:fs";
import { WaitingRoom, WorkerPool, workersFromEnv, type WorkerConfig } from "../src";

function localWorker(): WorkerConfig[] {
  const env = readFileSync(new URL("../../.env", import.meta.url), "utf8");
  const token = env.match(/^WORKER_TOKEN=(.+)$/m)?.[1]?.trim();
  if (!token) throw new Error("WORKER_TOKEN missing in .env");
  return [{ name: "office", kind: "own", url: "http://127.0.0.1:7100", token }];
}

function toneWav(sec = 2.5, sr = 16000): Uint8Array {
  const n = Math.floor(sec * sr);
  const buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8); buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    const env = 0.5 + 0.5 * Math.sin((2 * Math.PI * 3 * i) / sr);
    buf.writeInt16LE(Math.round(8000 * env * Math.sin((2 * Math.PI * 180 * i) / sr)), 44 + i * 2);
  }
  return new Uint8Array(buf);
}

const workers = process.env.MASCOTTE_WORKERS ? workersFromEnv() : localWorker();
const callers = Number(process.argv[2] ?? 4);
const face = new Uint8Array(readFileSync(process.env.SMOKE_FACE ?? "/home/khitan/age-progression-app/client/public/examples/3.jpg"));

const pool = new WorkerPool(workers, { onPressure: (s) => console.log(`  pressure: ${s.freeSlots} free slot(s) left -> would start a cloud worker`) });
const room = new WaitingRoom(pool, { pollMs: 1000 });
pool.start();
await pool.refresh();
console.log("pool:", pool.snapshot().workers.map((w) => `${w.name} ready=${w.ready} free=${w.free}`).join(", "));

const t0 = Date.now();
const results = await Promise.all(Array.from({ length: callers }, async (_, i) => {
  const tier = i === callers - 1 ? "paid" : "free";
  const call = await room.enter(face, "image/jpeg", {
    tier, onPosition: (p) => console.log(`  caller ${i + 1} (${tier}) waiting, place ${p}`),
  });
  const waited = (Date.now() - t0) / 1000;
  const res = await pool.renderClip(call.id, toneWav(), 1, 0);
  const kb = (await res.arrayBuffer()).byteLength >> 10;
  console.log(`  caller ${i + 1}: ${call.worker}, got a call after ${waited.toFixed(1)}s, clip ${res.status} ${kb} KB`);
  // Hold the call briefly, as a real conversation would, then hang up.
  await new Promise((r) => setTimeout(r, 1500));
  await pool.closeCall(call.id);
  room.nudge();
  return res.status === 200;
}));

pool.stop();
room.stop();
console.log(results.every(Boolean) ? "DEMO PASS" : "DEMO FAIL");
process.exit(results.every(Boolean) ? 0 : 1);
