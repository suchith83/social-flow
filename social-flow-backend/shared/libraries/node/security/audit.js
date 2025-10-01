/**
 * audit.js
 *
 * Security audit logger for important events:
 * - logAuthEvent(userId, action, details)
 * - logConfigChange(actor, details)
 *
 * Writes NDJSON lines to file and exposes an event emitter hook for remote shipping.
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const config = require('./config');
const logger = require('./logger');

const emitter = new EventEmitter();
const auditFile = path.resolve(config.AUDIT.logFile);

// ensure dir exists
try {
  const dir = path.dirname(auditFile);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
} catch (e) {
  // ignore
}

function _write(entry) {
  if (!config.AUDIT.enabled) return;
  const line = JSON.stringify(entry) + '\n';
  try {
    fs.appendFileSync(auditFile, line, { encoding: 'utf8', mode: 0o600 });
  } catch (e) {
    logger.warn({ e }, 'Failed to write audit log');
  }
  emitter.emit('entry', entry);
}

function logAuthEvent({ userId = null, remoteIp = null, action = '', success = true, details = {} } = {}) {
  const entry = {
    ts: new Date().toISOString(),
    type: 'auth',
    userId,
    remoteIp,
    action,
    success,
    details,
  };
  _write(entry);
  return entry;
}

function logConfigChange({ actor = null, changes = {}, details = {} } = {}) {
  const entry = {
    ts: new Date().toISOString(),
    type: 'config_change',
    actor,
    changes,
    details,
  };
  _write(entry);
  return entry;
}

function onEntry(fn) {
  emitter.on('entry', fn);
}

module.exports = {
  logAuthEvent,
  logConfigChange,
  onEntry,
  emitter,
};
