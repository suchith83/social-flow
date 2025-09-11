/**
 * common/libraries/node/database/utils.js
 * Small utilities: exponential backoff, sleep, camel/snake conversions, safe identifiers.
 */

const crypto = require('crypto');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoff(attempt, base = 100, cap = 2000) {
  // Exponential backoff with full jitter
  const expo = Math.min(cap, base * Math.pow(2, attempt));
  return Math.floor(Math.random() * expo);
}

function isTransientError(err) {
  // Lightweight heuristic: check common transient error codes from pg/mysql
  if (!err || !err.code) return false;
  const transientPg = [
    '57P01', // admin_shutdown
    '57P02', // crash_shutdown
    '57P03', // cannot_connect_now
    '08006', // connection_failure
    '08001', // sqlclient_unable_to_establish_sqlconnection
    '53300', // too_many_connections
  ];
  const transientMySQL = [
    'PROTOCOL_CONNECTION_LOST',
    'ER_LOCK_DEADLOCK',
    'ER_CON_COUNT_ERROR',
    'ETIMEDOUT',
    'ECONNRESET',
    'ECONNREFUSED',
  ];

  return transientPg.includes(err.code) || transientMySQL.includes(err.code) || transientMySQL.includes(err.errno) || false;
}

function paramPlaceholder(client, idx) {
  // Postgres uses $1, $2... ; MySQL uses ?
  if (client === 'pg') {
    return `$${idx}`;
  }
  return `?`;
}

function safeIdent(name) {
  // Minimal identifier sanitizer — for columns/table names. Use with caution.
  if (typeof name !== 'string') throw new Error('Identifier must be string');
  if (!/^[a-zA-Z0-9_]+$/.test(name)) {
    throw new Error(`Unsafe identifier: ${name}`);
  }
  return name;
}

function genRandomId(len = 16) {
  return crypto.randomBytes(len).toString('hex');
}

module.exports = {
  sleep,
  backoff,
  isTransientError,
  paramPlaceholder,
  safeIdent,
  genRandomId,
};
