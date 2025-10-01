/**
 * localFsAdapter.js
 * Simple adapter that stores files on local filesystem under basePath.
 * Designed for development & testing.
 *
 * Important:
 * - Uses safeJoin to prevent path traversal
 * - Stores metadata as .meta JSON sidecar files
 * - Streams for upload/download
 */

const fs = require('fs');
const path = require('path');
const { pipeline } = require('stream');
const { promisify } = require('util');
const Adapter = require('../adapter');
const config = require('../config');
const utils = require('../utils');
const logger = require('../logger');

const pipe = promisify(pipeline);

class LocalFsAdapter extends Adapter {
  constructor(opts = {}) {
    super(opts);
    this.basePath = opts.basePath || config.LOCAL.basePath;
  }

  async init() {
    utils.ensureDirSync(this.basePath);
    logger.info({ basePath: this.basePath }, 'LocalFsAdapter initialized');
  }

  _metaPath(filepath) {
    return `${filepath}.meta.json`;
  }

  async upload({ key, stream, buffer, size, contentType, metadata = {} }) {
    if (!key) throw new Error('upload requires key');
    const dest = utils.safeJoin(this.basePath, key);
    const dir = path.dirname(dest);
    utils.ensureDirSync(dir);

    // write stream or buffer
    if (buffer) {
      await fs.promises.writeFile(dest, buffer, { mode: 0o640 });
    } else if (stream) {
      const ws = fs.createWriteStream(dest, { mode: 0o640 });
      await pipe(stream, ws);
    } else {
      throw new Error('upload requires stream or buffer');
    }

    const stat = await fs.promises.stat(dest);
    const etag = await this._computeEtag(dest);

    const meta = {
      key,
      size: stat.size,
      contentType: contentType || utils.guessMime(key),
      metadata,
      etag,
      lastModified: stat.mtime.toISOString(),
    };
    await fs.promises.writeFile(this._metaPath(dest), JSON.stringify(meta, null, 2), { mode: 0o600 });

    return { key, size: stat.size, etag, metadata: meta };
  }

  async _computeEtag(filepath) {
    // simple sha256 hex
    const { createReadStream } = require('fs');
    const stream = createReadStream(filepath);
    const hash = require('crypto').createHash('sha256');
    return await new Promise((resolve, reject) => {
      stream.on('data', (d) => hash.update(d));
      stream.on('end', () => resolve(hash.digest('hex')));
      stream.on('error', reject);
    });
  }

  async download({ key }) {
    const dest = utils.safeJoin(this.basePath, key);
    try {
      await fs.promises.access(dest, fs.constants.R_OK);
    } catch (e) {
      throw new Error('Not found');
    }
    const stat = await fs.promises.stat(dest);
    const rs = fs.createReadStream(dest);
    let metadata = null;
    try {
      const metaRaw = await fs.promises.readFile(this._metaPath(dest), 'utf8');
      metadata = JSON.parse(metaRaw);
    } catch (e) {
      metadata = null;
    }
    return { stream: rs, size: stat.size, contentType: metadata?.contentType || utils.guessMime(key), metadata };
  }

  async stat({ key }) {
    const dest = utils.safeJoin(this.basePath, key);
    try {
      const stat = await fs.promises.stat(dest);
      let metadata = {};
      try {
        const metaRaw = await fs.promises.readFile(this._metaPath(dest), 'utf8');
        metadata = JSON.parse(metaRaw);
      } catch (e) {}
      return { key, size: stat.size, lastModified: stat.mtime, metadata };
    } catch (e) {
      throw new Error('Not found');
    }
  }

  async delete({ key }) {
    const dest = utils.safeJoin(this.basePath, key);
    try {
      await fs.promises.unlink(dest);
      const metaPath = this._metaPath(dest);
      try { await fs.promises.unlink(metaPath); } catch (e) {}
      return { deleted: true };
    } catch (e) {
      if (e.code === 'ENOENT') return { deleted: false };
      throw e;
    }
  }

  async list(prefix = '', opts = {}) {
    // naive recursive list; for small dev datasets only
    const root = utils.safeJoin(this.basePath, prefix);
    const results = [];
    async function walk(dir) {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) {
          await walk(full);
        } else {
          if (full.endsWith('.meta.json')) continue;
          const rel = path.relative(this.basePath, full).replace(/\\/g, '/');
          const stat = await fs.promises.stat(full);
          results.push({ key: rel, size: stat.size, lastModified: stat.mtime });
        }
      }
    }
    await walk.call(this, root);
    return results;
  }

  // presign not supported for local adapter; can generate local temp URLs in app if needed
  async presignGet(key, opts = {}) {
    throw new Error('presign not supported for local adapter');
  }

  async presignPut(key, opts = {}) {
    throw new Error('presign not supported for local adapter');
  }
}

module.exports = LocalFsAdapter;
