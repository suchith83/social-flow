/**
 * Generic helper utilities for tests.
 */

export function randomString(length = 8) {
  return Math.random().toString(36).substring(2, 2 + length);
}

export function timestamp() {
  return new Date().toISOString();
}

/**
 * Sleep helper (ms)
 */
export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
