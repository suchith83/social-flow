/**
 * deep.js
 *
 * Deep merge, clone, and path helpers.
 * - deepClone(obj)
 * - deepMerge(target, source, opts) — merges nested objects/arrays with options
 * - getPath(obj, path, defaultValue)
 * - setPath(obj, path, value)
 *
 * Implementation notes:
 * - Avoids prototype pollution by skipping __proto__ and constructor keys
 * - Cloning handles Dates, RegExps, Buffers, plain objects and arrays
 */

function isPlainObject(obj) {
  return Object.prototype.toString.call(obj) === '[object Object]';
}

function deepClone(value, _seen = new WeakMap()) {
  if (value === null || typeof value !== 'object') return value;
  if (_seen.has(value)) return _seen.get(value);
  if (Array.isArray(value)) {
    const arr = [];
    _seen.set(value, arr);
    for (const v of value) arr.push(deepClone(v, _seen));
    return arr;
  }
  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof RegExp) return new RegExp(value.source, value.flags);
  if (Buffer.isBuffer(value)) return Buffer.from(value);
  if (isPlainObject(value)) {
    const out = {};
    _seen.set(value, out);
    for (const k of Object.keys(value)) {
      if (k === '__proto__' || k === 'constructor') continue;
      out[k] = deepClone(value[k], _seen);
    }
    return out;
  }
  // other types: return as-is
  return value;
}

function deepMerge(target = {}, source = {}, { arrayMerge = 'replace' } = {}) {
  // arrayMerge: 'concat' | 'replace' | 'unique'
  const out = deepClone(target);
  if (!isPlainObject(source)) return out;
  for (const key of Object.keys(source)) {
    if (key === '__proto__' || key === 'constructor') continue;
    const sVal = source[key];
    const tVal = out[key];
    if (Array.isArray(sVal) && Array.isArray(tVal)) {
      if (arrayMerge === 'concat') out[key] = tVal.concat(deepClone(sVal));
      else if (arrayMerge === 'unique') out[key] = Array.from(new Set(tVal.concat(sVal)));
      else out[key] = deepClone(sVal);
    } else if (isPlainObject(sVal) && isPlainObject(tVal)) {
      out[key] = deepMerge(tVal, sVal, { arrayMerge });
    } else {
      out[key] = deepClone(sVal);
    }
  }
  return out;
}

function getPath(obj, path, defaultValue = undefined) {
  if (!path) return defaultValue;
  const parts = Array.isArray(path) ? path : String(path).split(/[.[\]]+/).filter(Boolean);
  let curr = obj;
  for (const p of parts) {
    if (curr == null) return defaultValue;
    curr = curr[p];
  }
  return curr === undefined ? defaultValue : curr;
}

function setPath(obj, path, value) {
  const parts = Array.isArray(path) ? path : String(path).split(/[.[\]]+/).filter(Boolean);
  let curr = obj;
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (i === parts.length - 1) {
      curr[p] = value;
    } else {
      if (!isPlainObject(curr[p])) curr[p] = {};
      curr = curr[p];
    }
  }
  return obj;
}

module.exports = {
  deepClone,
  deepMerge,
  getPath,
  setPath,
};
