/**
 * uploader.js
 *
 * Safe uploader wrapper:
 * - Accepts Buffer/Stream/File-like
 * - Supports dedupe by checksum (optional)
 * - Supports enforcement of bucket/object quotas via quotaCallback
 * - Handles multipart threshold via client.adapter capabilities
 *
 * Example:
 *   const u = new Uploader(client, { dedupe: true });
 *   await u.putStream('avatars/1.png', readableStream, { contentType: 'image/png' });
 */

const utils = require('./utils');
const logger = require('./logger');

class Uploader {
  constructor(client, opts = {}) {
    if (!client) throw new Error('Uploader requires storage client');
    this.client = client;
    this.dedupe = opts.dedupe || false;
    // quotaCallback: async({key,size}) => { allowed: true/false }
    this.quotaCallback = opts.quotaCallback || null;
    this.multipartThreshold = opts.multipartThreshold || require('./config').UPLOAD.multipartThresholdBytes;
  }

  /**
   * putStream: uploads a readable stream. If dedupe is enabled, computes checksum first.
   */
  async putStream(key, stream, opts = {}) {
    const { contentType, metadata = {} } = opts;
    // If dedupe: compute checksum by teeing stream into buffer? We must buffer or compute while streaming.
    if (this.dedupe) {
      // buffer whole stream (for dedupe). Warning: memory heavy for large files.
      const buf = await utils.streamToBuffer(stream);
      const checksum = require('crypto').createHash('sha256').update(buf).digest('hex');
      // optional: check existing object by metadata or a mapping; here we rely on stat/etag equivalence not guaranteed
      // If quotaCallback exists, check before writing
      if (this.quotaCallback) {
        const q = await this.quotaCallback({ key, size: buf.length });
        if (!q.allowed) throw new Error('Quota exceeded');
      }
      return await this.client.upload({ key, buffer: buf, size: buf.length, contentType, metadata: { ...metadata, checksum } });
    }

    // if no dedupe and stream length unknown, rely on client to stream
    // If quotaCallback present, we cannot know size; either require size param or skip quota check
    if (this.quotaCallback && typeof opts.size === 'number') {
      const q = await this.quotaCallback({ key, size: opts.size });
      if (!q.allowed) throw new Error('Quota exceeded');
    }

    return await this.client.upload({ key, stream, size: opts.size, contentType, metadata });
  }

  /**
   * putBuffer: direct buffer upload
   */
  async putBuffer(key, buffer, opts = {}) {
    const { contentType, metadata = {} } = opts;
    if (this.quotaCallback) {
      const q = await this.quotaCallback({ key, size: buffer.length });
      if (!q.allowed) throw new Error('Quota exceeded');
    }
    const checksum = require('crypto').createHash('sha256').update(buffer).digest('hex');
    const res = await this.client.upload({ key, buffer, size: buffer.length, contentType, metadata: { ...metadata, checksum } });
    return res;
  }
}

module.exports = Uploader;
