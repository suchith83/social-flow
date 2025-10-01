/**
 * memoize.js
 *
 * Memoization helper for sync and async functions.
 * - Supports TTL-based cache invalidation
 * - Supports max size (LRU eviction)
 * - Supports custom cache key resolver
 *
 * Implementation uses a simple Map + order tracking for LRU (sufficient for moderate sizes).
 *
 * Usage:
 *   const mem = memoize(fn, { ttl: 1000, maxSize: 1000, key: (args)=> ... })
 */

function memoize(fn, { ttl = 0, maxSize = 1000, cacheKey = null } = {}) {
  const cache = new Map(); // key -> { value, exp, node }
  const order = []; // keys in insertion order for simple LRU

  function makeKey(args) {
    if (cacheKey) return cacheKey(args);
    try {
      return JSON.stringify(args);
    } catch (e) {
      // fallback to stringified inspection
      return String(args);
    }
  }

  function touchKey(key) {
    const idx = order.indexOf(key);
    if (idx !== -1) order.splice(idx, 1);
    order.push(key);
    // trim if necessary
    while (order.length > maxSize) {
      const k = order.shift();
      cache.delete(k);
    }
  }

  return function memoized(...args) {
    const key = makeKey(args);
    const now = Date.now();
    if (cache.has(key)) {
      const entry = cache.get(key);
      if (!entry.exp || entry.exp > now) {
        touchKey(key);
        return entry.value;
      } else {
        cache.delete(key);
        const idx = order.indexOf(key);
        if (idx !== -1) order.splice(idx, 1);
      }
    }
    const result = fn.apply(this, args);
    const isPromise = result && typeof result.then === 'function';
    if (isPromise) {
      // store promise immediately so concurrent callers share the same execution
      const p = result.then((v) => {
        cache.set(key, { value: Promise.resolve(v), exp: ttl ? now + ttl : 0 });
        touchKey(key);
        return v;
      }).catch((err) => {
        cache.delete(key);
        const idx = order.indexOf(key); if (idx !== -1) order.splice(idx, 1);
        throw err;
      });
      cache.set(key, { value: p, exp: ttl ? now + ttl : 0 });
      touchKey(key);
      return p;
    } else {
      cache.set(key, { value: result, exp: ttl ? now + ttl : 0 });
      touchKey(key);
      return result;
    }
  };
}

module.exports = memoize;
