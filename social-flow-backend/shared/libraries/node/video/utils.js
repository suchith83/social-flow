/**
 * Utility helpers for file management, backoff, runtime checks, temp files, and wrappers.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { promisify } = require('util');
const crypto = require('crypto');
const fse = require('fs-extra');
const config = require('./config');
const logger = require('./logger');

const mkdtemp = promisify(fs.mkdtemp);

async function createTempDir(prefix = 'video-', base = config.TEMP_DIR) {
  await fse.ensureDir(base);
  const dir = await mkdtemp(path.join(base, prefix));
  return dir;
}

async function removeTempDir(dir) {
  try {
    if (config.CLEANUP_TEMP) await fse.remove(dir);
  } catch (e) {
    logger.warn({ err: e, dir }, 'Failed to cleanup temp dir');
  }
}

function randomId(len = 12) {
  return crypto.randomBytes(Math.ceil(len / 2)).toString('hex').slice(0, len);
}

function exponentialBackoff(attempt = 0, base = config.TRANSCODE.baseBackoffMs, cap = 10_000) {
  const raw = Math.min(cap, base * Math.pow(2, attempt));
  // full jitter
  return Math.floor(Math.random() * raw);
}

function safeFilename(name = '') {
  return name.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 240);
}

function ensureDirSync(dir) {
  fse.ensureDirSync(dir);
}

module.exports = {
  createTempDir,
  removeTempDir,
  randomId,
  exponentialBackoff,
  safeFilename,
  ensureDirSync,
};
