/**
 * Utility helpers for monitoring
 * - timing helpers
 * - safe stringify
 * - convenience wrappers
 */

const ms = require('ms');

function nowIso() {
  return new Date().toISOString();
}

function durationToMs(durationStr) {
  // parse strings like "1s" "100ms" using ms package if available
  try {
    return ms(durationStr);
  } catch (e) {
    // fallback naive parse
    const num = parseFloat(durationStr);
    return Number.isNaN(num) ? null : num;
  }
}

function safeStringify(obj) {
  try {
    return JSON.stringify(obj);
  } catch (e) {
    const seen = new WeakSet();
    return JSON.stringify(obj, (k, v) => {
      if (typeof v === 'object' && v !== null) {
        if (seen.has(v)) return '[Circular]';
        seen.add(v);
      }
      if (typeof v === 'bigint') return v.toString();
      return v;
    });
  }
}

module.exports = {
  nowIso,
  durationToMs,
  safeStringify,
};
