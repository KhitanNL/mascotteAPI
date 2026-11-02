import type { WorkerConfig, WorkerKind } from "./pool";

/**
 * Workers from the environment, one per line or separated by ";":
 *
 *   MASCOTTE_WORKERS="office|own|http://127.0.0.1:7100|<token>; cloud-1|cloud|https://...|<token>"
 *
 * Cloud workers that the autoscaler starts are added at runtime instead.
 */
export function workersFromEnv(value = process.env.MASCOTTE_WORKERS ?? ""): WorkerConfig[] {
  return value
    .split(/[;\n]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((line) => {
      const [name, kind, url, token] = line.split("|").map((s) => s.trim());
      if (!name || !url || !token || (kind !== "own" && kind !== "cloud")) {
        throw new Error(`MASCOTTE_WORKERS: expected "name|own|url|token" or "name|cloud|url|token", got "${name ?? line}|…"`);
      }
      return { name, kind: kind as WorkerKind, url, token };
    });
}
