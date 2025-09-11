/**
 * env.js
 *
 * Safe environment helpers:
 * - getEnv(name, { required, default, parse })
 * - requireEnv(name) -> throws if missing
 * - loadDotenv(path) -> load .env with fallback
 *
 * Provides safe parsing for numbers/booleans and helpful errors.
 */

const fs = require('fs');

function parseValue(val, parse) {
  if (parse === 'int') return parseInt(val, 10);
  if (parse === 'float') return parseFloat(val);
  if (parse === 'bool') return ['1', 'true', 'yes', 'on'].includes(String(val).toLowerCase());
  if (typeof parse === 'function') return parse(val);
  return val;
}

function getEnv(name, opts = {}) {
  const { required = false, defaultValue = undefined, parse = null } = opts;
  const raw = process.env[name];
  if (raw === undefined || raw === null || raw === '') {
    if (required && defaultValue === undefined) {
      throw new Error(`Environment variable ${name} is required`);
    }
    return defaultValue;
  }
  return parseValue(raw, parse);
}

function requireEnv(name, parse = null) {
  return getEnv(name, { required: true, parse });
}

function loadDotenv(path = '.env') {
  try {
    const dotenv = require('dotenv');
    if (fs.existsSync(path)) {
      const res = dotenv.config({ path });
      return res;
    }
    return null;
  } catch (e) {
    // silent if dotenv not installed
    return null;
  }
}

module.exports = {
  getEnv,
  requireEnv,
  loadDotenv,
};
