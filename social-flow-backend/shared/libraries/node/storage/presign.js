/**
 * presign.js
 * Convenience helpers around Client.presignPut/Get
 */

const config = require('./config');
const logger = require('./logger');

class Presign {
  constructor(client) {
    if (!client) throw new Error('Presign requires client');
    this.client = client;
    this.defaultExpires = config.PRESIGN.urlExpiresSec;
  }

  async getUrl(key, opts = {}) {
    return await this.client.presignGet(key, { expiresSec: opts.expiresSec || this.defaultExpires });
  }

  async putUrl(key, opts = {}) {
    return await this.client.presignPut(key, { contentType: opts.contentType, expiresSec: opts.expiresSec || this.defaultExpires });
  }
}

module.exports = Presign;
