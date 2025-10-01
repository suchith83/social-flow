/**
 * Password hashing and verification with bcrypt
 */

const bcrypt = require('bcrypt');
const config = require('./config');

module.exports = {
  hashPassword: async (password) => {
    return await bcrypt.hash(password, config.security.saltRounds);
  },

  verifyPassword: async (password, hash) => {
    return await bcrypt.compare(password, hash);
  },
};
