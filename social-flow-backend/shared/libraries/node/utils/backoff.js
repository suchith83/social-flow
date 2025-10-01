/**
 * backoff.js
 *
 * Implements multiple backoff strategies:
 * - exponentialBackoff(attempt, base, cap, jitterFull)
 * - linearBackoff(...)
 * - fibonacciBackoff(...)
 *
 * Returns ms to wait.
 *
 * Also includes backoffWithJitter helper (full jitter per AWS architecture).
 */

function exponentialBackoff(attempt = 0, base = 100, cap = 2000, fullJitter = true) {
  // attempt 0 => base
  const raw = Math.min(cap, base * Math.pow(2, attempt));
  if (!fullJitter) return raw;
  return Math.floor(Math.random() * raw);
}

function linearBackoff(attempt = 0, base = 100, cap = 2000) {
  const raw = Math.min(cap, base * (attempt + 1));
  return raw;
}

function fibonacciBackoff(attempt = 0, base = 100, cap = 2000, fullJitter = true) {
  // fib sequence
  const fib = (n) => {
    let a = 1, b = 1;
    for (let i = 2; i <= n; i++) {
      const c = a + b; a = b; b = c;
    }
    return n === 0 ? 1 : (n === 1 ? 1 : b);
  };
  const raw = Math.min(cap, base * fib(attempt));
  return fullJitter ? Math.floor(Math.random() * raw) : raw;
}

module.exports = {
  exponentialBackoff,
  linearBackoff,
  fibonacciBackoff,
};
