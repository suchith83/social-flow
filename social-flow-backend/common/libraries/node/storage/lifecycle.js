/**
 * lifecycle.js
 *
 * Utilities to manage lifecycle rules such as:
 * - garbage-collect unreferenced objects (based on predicate)
 * - expire objects older than N days
 * - compute bucket usage (scan)
 *
 * WARNING: these operations can be expensive; use with care.
 */

const logger = require('./logger');

class Lifecycle {
  constructor(client) {
    if (!client) throw new Error('Lifecycle requires storage client');
    this.client = client;
  }

  /**
   * scanUsage: compute total bytes and count under prefix
   * returns { bytes, count }
   */
  async scanUsage(prefix = '') {
    let totalBytes = 0;
    let count = 0;
    for await (const obj of this.client.list(prefix)) {
      totalBytes += obj.size || 0;
      count += 1;
    }
    return { bytes: totalBytes, count };
  }

  /**
   * expireOlderThan: delete objects older than cutoffDate (Date or ISO)
   * returns list of deleted keys (be careful)
   */
  async expireOlderThan(prefix = '', cutoffDate) {
    const cutoff = cutoffDate instanceof Date ? cutoffDate : new Date(cutoffDate);
    const deleted = [];
    for await (const obj of this.client.list(prefix)) {
      const last = new Date(obj.lastModified || obj.LastModified || obj.last_modified || Date.now());
      if (last < cutoff) {
        try {
          await this.client.delete({ key: obj.key });
          deleted.push(obj.key);
        } catch (e) {
          logger.warn({ err: e, key: obj.key }, 'Failed to delete during expireOlderThan');
        }
      }
    }
    return deleted;
  }

  /**
   * gcUnreferenced: find objects for which predicate returns true and delete them.
   * predicate: async (obj) => boolean
   */
  async gcUnreferenced(prefix = '', predicate) {
    if (typeof predicate !== 'function') throw new Error('predicate required');
    const deleted = [];
    for await (const obj of this.client.list(prefix)) {
      try {
        const should = await predicate(obj);
        if (should) {
          await this.client.delete({ key: obj.key });
          deleted.push(obj.key);
        }
      } catch (e) {
        logger.warn({ err: e, key: obj.key }, 'gcUnreferenced predicate error');
      }
    }
    return deleted;
  }
}

module.exports = Lifecycle;
