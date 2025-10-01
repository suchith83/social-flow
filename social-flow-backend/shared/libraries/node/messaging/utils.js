/**
 * common/libraries/node/messaging/utils.js
 * Utility helpers: backoff, idempotency keys, safe JSON parse/stringify, timers.
 */

const crypto = require('crypto');

function genId(prefix = '', len = 12) {
  return prefix + crypto.randomBytes(len).toString('hex');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoffMs(attempt, base = 100, cap = 2000) {
  // Exponential backoff with full jitter
  const exp = Math.min(cap, base * Math.pow(2, attempt));
  return Math.floor(Math.random() * exp);
}

function safeJsonParse(s) {
  try {
    return JSON.parse(s);
  } catch (e) {
    return null;
  }
}

function safeJsonStringify(obj) {
  try {
    return JSON.stringify(obj);
  } catch (e) {
    // fallback for circular refs
    return JSON.stringify(serializeSafe(obj));
  }
}

function serializeSafe(obj) {
  const seen = new WeakSet();
  return JSON.parse(
    JSON.stringify(obj, function (key, value) {
      if (typeof value === 'object' && value !== null) {
        if (seen.has(value)) return '[Circular]';
        seen.add(value);
      }
      return value;
    })
  );
}

/**
 * Attach idempotency metadata to message headers.
 * Different brokers use different header shapes — we keep normalized header keys.
 */
function attachIdempotency(headers = {}, id) {
  return { ...headers, 'x-msg-idempotency-key': id };
}

module.exports = {
  genId,
  sleep,
  backoffMs,
  safeJsonParse,
  safeJsonStringify,
  serializeSafe,
  attachIdempotency,
};
