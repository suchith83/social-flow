/**
 * Utility functions for cryptographic operations
 */

const crypto = require('crypto');

module.exports = {
  generateRandomToken: (size = 48) => crypto.randomBytes(size).toString('hex'),

  hashSHA256: (input) =>
    crypto.createHash('sha256').update(input).digest('hex'),

  constantTimeCompare: (a, b) => {
    // Prevent timing attacks
    const aLen = Buffer.byteLength(a);
    const bLen = Buffer.byteLength(b);
    if (aLen !== bLen) return false;
    return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
  },

  base64URLEncode: (input) =>
    input.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
};
