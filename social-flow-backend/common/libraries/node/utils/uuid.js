/**
 * uuid.js
 *
 * Lightweight UUID v4 generator using crypto.randomBytes to avoid dependency.
 * Also provides simple id generators (shortId).
 */

const crypto = require('crypto');

function uuidv4() {
  const b = crypto.randomBytes(16);
  // per RFC4122 v4
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const hex = b.toString('hex');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20,32)}`;
}

function shortId(prefix = '', len = 8) {
  const b = crypto.randomBytes(Math.ceil(len/2)).toString('hex').slice(0, len);
  return prefix ? `${prefix}-${b}` : b;
}

module.exports = {
  uuidv4,
  shortId,
};
