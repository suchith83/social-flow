/**
 * guards.js
 *
 * Assertion and guard helpers to make runtime checks expressive.
 * - assert(condition, message)
 * - assertType(value, type, message)
 * - mustGet(obj, path) -> throws if missing
 * - isEmpty(value)
 *
 * These helpers throw descriptive errors to surface programmer mistakes early.
 */

function assert(cond, msg = 'Assertion failed') {
  if (!cond) {
    const e = new Error(msg);
    e.name = 'AssertionError';
    throw e;
  }
}

function assertType(value, type, msg) {
  const got = typeof value;
  if (type === 'array') {
    if (!Array.isArray(value)) throw new TypeError(msg || `Expected array but got ${got}`);
  } else if (type === 'buffer') {
    if (!Buffer.isBuffer(value)) throw new TypeError(msg || `Expected buffer but got ${got}`);
  } else if (type === 'date') {
    if (!(value instanceof Date)) throw new TypeError(msg || `Expected Date but got ${got}`);
  } else {
    if (got !== type) throw new TypeError(msg || `Expected ${type} but got ${got}`);
  }
}

function mustGet(obj, path) {
  const { getPath } = require('./deep');
  const v = getPath(obj, path, undefined);
  if (v === undefined) throw new Error(`Missing required value at path: ${path}`);
  return v;
}

function isEmpty(v) {
  if (v == null) return true;
  if (typeof v === 'string') return v.length === 0;
  if (Array.isArray(v)) return v.length === 0;
  if (typeof v === 'object') return Object.keys(v).length === 0;
  return false;
}

module.exports = {
  assert,
  assertType,
  mustGet,
  isEmpty,
};
