/**
 * multipart.js
 *
 * High-level multipart manager for adapters that support multipart (e.g., S3).
 * Offers:
 *  - startMultipart(key) -> { uploadId }
 *  - uploadPart(key, uploadId, partNumber, stream|buffer) -> etag
 *  - completeMultipart(key, uploadId, parts)
 *  - abortMultipart(key, uploadId)
 *
 * For S3Adapter we rely on client's multipart APIs.
 *
 * This module is minimal; many apps prefer the SDK's high-level multipart helper.
 */

const logger = require('./logger');
const utils = require('./utils');

class Multipart {
  constructor(adapter) {
    this.adapter = adapter;
    if (!this.adapter) throw new Error('Multipart requires adapter');
  }

  // Delegates to adapter if implemented
  async startMultipart(opts) {
    if (typeof this.adapter.startMultipart === 'function') return await this.adapter.startMultipart(opts);
    throw new Error('startMultipart not implemented by adapter');
  }

  async uploadPart(opts) {
    if (typeof this.adapter.uploadPart === 'function') return await this.adapter.uploadPart(opts);
    throw new Error('uploadPart not implemented by adapter');
  }

  async completeMultipart(opts) {
    if (typeof this.adapter.completeMultipart === 'function') return await this.adapter.completeMultipart(opts);
    throw new Error('completeMultipart not implemented by adapter');
  }

  async abortMultipart(opts) {
    if (typeof this.adapter.abortMultipart === 'function') return await this.adapter.abortMultipart(opts);
    throw new Error('abortMultipart not implemented by adapter');
  }
}

module.exports = Multipart;
