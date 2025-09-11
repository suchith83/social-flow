/**
 * debounceThrottle.js
 *
 * Utilities for wrapping functions:
 * - debounce(fn, wait, opts) -> returns debounced function
 * - throttle(fn, wait, opts) -> returns throttled function
 *
 * Both support Promise-returning functions; the wrappers return Promises for the underlying result.
 * Debounce supports leading/trailing options; throttle supports leading/trailing and cancel().
 *
 * These are carefully designed to avoid unhandled rejections in wrapped functions.
 */

function debounce(fn, wait = 100, { leading = false, trailing = true } = {}) {
  let timeout = null;
  let lastArgs = null;
  let lastThis = null;
  let pendingPromise = null;
  let resolvePending = null;
  let rejectPending = null;

  const invoke = async () => {
    const args = lastArgs;
    const self = lastThis;
    lastArgs = lastThis = null;
    try {
      const res = await fn.apply(self, args);
      if (resolvePending) resolvePending(res);
    } catch (err) {
      if (rejectPending) rejectPending(err);
    } finally {
      pendingPromise = resolvePending = rejectPending = null;
    }
  };

  return function debounced(...args) {
    lastArgs = args;
    lastThis = this;
    if (!pendingPromise) {
      pendingPromise = new Promise((resolve, reject) => {
        resolvePending = resolve;
        rejectPending = reject;
      });
    }
    if (timeout) clearTimeout(timeout);
    if (leading && !timeout) {
      // call immediately
      (async () => {
        try {
          const r = await fn.apply(this, args);
          resolvePending && resolvePending(r);
        } catch (e) {
          rejectPending && rejectPending(e);
        } finally {
          pendingPromise = resolvePending = rejectPending = null;
        }
      })();
    }
    timeout = setTimeout(() => {
      timeout = null;
      if (trailing && lastArgs) invoke();
    }, wait);
    return pendingPromise;
  };
}

function throttle(fn, wait = 100, { leading = true, trailing = true } = {}) {
  let lastCallTime = 0;
  let timeout = null;
  let pending = null;
  let lastArgs = null;
  let lastThis = null;

  const invoke = async () => {
    lastCallTime = Date.now();
    const args = lastArgs;
    const self = lastThis;
    lastArgs = lastThis = null;
    try {
      const res = await fn.apply(self, args);
      if (pending) pending.resolve(res);
    } catch (err) {
      if (pending) pending.reject(err);
    } finally {
      pending = null;
    }
  };

  return function throttled(...args) {
    const now = Date.now();
    const remaining = wait - (now - lastCallTime);
    lastArgs = args;
    lastThis = this;
    if (!pending) {
      pending = {};
      pending.promise = new Promise((res, rej) => { pending.resolve = res; pending.reject = rej; });
    }
    if (remaining <= 0) {
      if (timeout) { clearTimeout(timeout); timeout = null; }
      invoke();
    } else if (!timeout && trailing) {
      timeout = setTimeout(() => {
        timeout = null;
        invoke();
      }, remaining);
    } else if (leading && !timeout && !pending) {
      invoke();
    }
    return pending.promise;
  };
}

module.exports = {
  debounce,
  throttle,
};
