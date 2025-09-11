/**
 * Security Audit Logger
 * Logs auth events (login, logout, failed attempts, etc.)
 */

const fs = require('fs');
const path = require('path');
const config = require('./config');

const logPath = path.resolve(config.audit.logFile);

module.exports = {
  logEvent: (eventType, details = {}) => {
    const entry = {
      timestamp: new Date().toISOString(),
      eventType,
      details,
    };

    const logLine = JSON.stringify(entry) + '\n';
    fs.appendFileSync(logPath, logLine, { encoding: 'utf8' });
  },
};
