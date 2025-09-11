/**
 * storage/utils.js
 * Utility helpers: backoff, streaming helpers, path helpers, mime type guessing, checksum
 */

const crypto = require('crypto');
const path = require('path');
const fs = require('fs');
const { pipeline } = require('stream');
const { promisify } = require('util');

const pipe = promisify(pipeline);

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function exponentialBackoff(attempt, base = 200, cap = 2000) {
  const expo = Math.min(cap, base * Math.pow(2, attempt));
  return Math.floor(Math.random() * expo);
}

function ensureDirSync(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true, mode: 0o750 });
}

/**
 * computeSHA256Stream: computes hex sha256 for a stream
 */
async function computeSHA256Stream(stream) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    stream.on('data', (d) => hash.update(d));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

/**
 * safeJoin: join a base path and user-provided key while preventing traversal
 */
function safeJoin(base, key) {
  const cleanKey = key.replace(/(\.\.(\/|\\|$))+/g, ''); // naive sanitize
  const joined = path.join(base, cleanKey);
  if (!joined.startsWith(path.resolve(base))) {
    throw new Error('Invalid storage key');
  }
  return joined;
}

/**
 * streamToBuffer (for small streams)
 */
async function streamToBuffer(stream) {
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  return Buffer.concat(chunks);
}

/**
 * guessMime: basic MIME guess using extension; lightweight to avoid extra dependency
 */
function guessMime(filename) {
  const ext = path.extname(filename || '').toLowerCase();
  const map = {
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.json': 'application/json',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.mp4': 'video/mp4',
    '.pdf': 'application/pdf',
  };
  return map[ext] || 'application/octet-stream';
}

module.exports = {
  sleep,
  exponentialBackoff,
  ensureDirSync,
  computeSHA256Stream,
  safeJoin,
  streamToBuffer,
  guessMime,
  pipe,
};
