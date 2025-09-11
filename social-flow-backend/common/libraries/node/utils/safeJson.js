/**
 * safeJson.js
 *
 * Robust JSON helpers:
 * - safeStringify(obj, replacer, space) -> avoids crash on circular refs by replacing with "[Circular]"
 * - safeParse(str, fallback) -> returns fallback on parse error
 *
 * Useful when logging unknown objects or reading external payloads.
 */

function safeStringify(obj, replacer = null, space = 2) {
  const seen = new WeakSet();
  return JSON.stringify(obj, function (k, v) {
    if (typeof v === 'object' && v !== null) {
      if (seen.has(v)) return '[Circular]';
      seen.add(v);
    }
    if (typeof v === 'bigint') return v.toString();
    return v;
  }, space);
}

function safeParse(str, fallback = null) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return fallback;
  }
}

module.exports = {
  safeStringify,
  safeParse,
};
