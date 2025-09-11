/**
 * retry.js
 *
 * Generic retry helper with support for:
 * - attempts, base backoff, max delay, jitter modes (none, full)
 * - conditional retry via shouldRetry(err, attempt)
 * - onRetry callback hook
 * - cancellation via AbortSignal
 *
 * Example:
 *   await retry(async () => await call(), { attempts: 5, baseMs: 100, jitter: 'full' });
 */

const backoffUtil = require('./backoff');

async function retry(fn, opts = {}) {
  const {
    attempts = 3,
    baseMs = 100,
    maxMs = 2000,
    jitter = 'full', // 'none' | 'full'
    shouldRetry = (err) => true,
    onRetry = null,
    signal = null,
  } = opts;

  let attempt = 0;
  let lastErr = null;
  while (attempt < attempts) {
    if (signal && signal.aborted) throw new Error('Aborted');
    try {
      const res = await fn(attempt);
      return res;
    } catch (err) {
      lastErr = err;
      attempt++;
      if (attempt >= attempts || !shouldRetry(err, attempt)) break;
      if (onRetry) {
        try { onRetry(err, attempt); } catch (_) {}
      }
      // compute backoff
      const delay = backoffUtil.exponentialBackoff(attempt - 1, baseMs, maxMs, jitter === 'full');
      await new Promise((res, rej) => {
        const to = setTimeout(res, delay);
        if (signal) signal.addEventListener('abort', () => { clearTimeout(to); rej(new Error('Aborted')); });
      });
    }
  }
  const e = new Error('Retries exhausted');
  e.cause = lastErr;
  throw e;
}

module.exports = retry;
