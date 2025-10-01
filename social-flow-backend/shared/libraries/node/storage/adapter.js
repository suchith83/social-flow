/**
 * storage/adapter.js
 *
 * Abstract storage adapter interface. Concrete adapters should implement:
 * - init()
 * - upload({ key, stream|buffer, size, contentType, metadata }) -> { key, size, etag, checksum }
 * - download({ key }) -> { stream, size, contentType, metadata }
 * - stat({ key }) -> { key, size, etag, metadata, lastModified }
 * - delete({ key }) -> { deleted: true }
 * - list(prefix) -> async iterable or array of { key, size, ... }
 * - presignGet/Put (optional)
 *
 * This is just a guideline (not enforced) but Client will expect these methods.
 */

class Adapter {
  constructor(opts = {}) {
    this.opts = opts;
  }

  async init() {
    throw new Error('init() not implemented');
  }

  async upload(opts) {
    throw new Error('upload() not implemented');
  }

  async download(opts) {
    throw new Error('download() not implemented');
  }

  async stat(opts) {
    throw new Error('stat() not implemented');
  }

  async delete(opts) {
    throw new Error('delete() not implemented');
  }

  async list(prefix, opts = {}) {
    throw new Error('list() not implemented');
  }

  async presignGet(key, opts = {}) {
    throw new Error('presignGet() not implemented');
  }

  async presignPut(key, opts = {}) {
    throw new Error('presignPut() not implemented');
  }
}

module.exports = Adapter;
