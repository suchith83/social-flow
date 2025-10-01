/**
 * JWT authentication and authorization helper
 */

const jwt = require('jsonwebtoken');
const config = require('./config');

module.exports = {
  sign: (payload, options = {}) =>
    jwt.sign(payload, config.jwt.secret, {
      expiresIn: config.jwt.expiresIn,
      issuer: config.jwt.issuer,
      ...options,
    }),

  verify: (token) => {
    try {
      return jwt.verify(token, config.jwt.secret, {
        issuer: config.jwt.issuer,
      });
    } catch (err) {
      throw new Error('Invalid or expired token');
    }
  },

  decode: (token) => jwt.decode(token, { complete: true }),
};
