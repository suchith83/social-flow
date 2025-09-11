/**
 * time.js
 *
 * Time helpers for high-resolution timing and human-readable conversions.
 * - hrTimeMs() -> number ms high-resolution
 * - nowIso() -> ISO timestamp
 * - measureAsync(fn) -> { value, elapsedMs }
 * - sleep(ms)
 *
 * Uses process.hrtime.bigint() for accuracy.
 */

function hrTimeMs() {
  if (typeof process !== 'undefined' && process.hrtime && process.hrtime.bigint) {
    const ns = process.hrtime.bigint();
    return Number(ns / BigInt(1e6));
  }
  return Date.now();
}

function nowIso() {
  return new Date().toISOString();
}

async function measureAsync(fn) {
  const start = hrTimeMs();
  const value = await fn();
  const end = hrTimeMs();
  return { value, elapsedMs: end - start };
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

module.exports = {
  hrTimeMs,
  nowIso,
  measureAsync,
  sleep,
};
