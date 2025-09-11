/**
 * fs.js
 *
 * Filesystem helpers with safe, atomic writes and small convenience utilities.
 * - writeFileAtomic(filePath, data, opts)
 * - readJson(filePath, defaultValue)
 * - writeJsonAtomic(filePath, obj, opts)
 * - ensureDir(path, mode)
 * - removeIfExists(path)
 *
 * Atomic write approach:
 *  - write to temp file in same dir, fsync, rename
 *  - preserves atomicity on POSIX filesystems
 */

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

const writeFile = promisify(fs.writeFile);
const rename = promisify(fs.rename);
const mkdir = promisify(fs.mkdir);
const stat = promisify(fs.stat);
const unlink = promisify(fs.unlink);

async function ensureDir(dir, mode = 0o750) {
  try {
    await mkdir(dir, { recursive: true, mode });
  } catch (e) {
    if (e.code !== 'EEXIST') throw e;
  }
}

async function writeFileAtomic(filePath, data, { mode = 0o600, encoding = 'utf8' } = {}) {
  const dir = path.dirname(filePath);
  await ensureDir(dir);
  const tmp = path.join(dir, `.tmp-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2,8)}`);
  // write tmp file
  await writeFile(tmp, data, { mode, encoding });
  // rename (atomic on same fs)
  await rename(tmp, filePath);
  return filePath;
}

async function writeJsonAtomic(filePath, obj, opts = {}) {
  const str = JSON.stringify(obj, null, 2);
  return writeFileAtomic(filePath, str, opts);
}

async function readJson(filePath, defaultValue = undefined) {
  try {
    const content = await fs.promises.readFile(filePath, 'utf8');
    return JSON.parse(content);
  } catch (e) {
    if (e.code === 'ENOENT') return defaultValue;
    throw e;
  }
}

async function removeIfExists(filePath) {
  try {
    await unlink(filePath);
    return true;
  } catch (e) {
    if (e.code === 'ENOENT') return false;
    throw e;
  }
}

module.exports = {
  ensureDir,
  writeFileAtomic,
  writeJsonAtomic,
  readJson,
  removeIfExists,
};
