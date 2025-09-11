/**
 * Entry point for common/libraries/node/monitoring
 * Exports the major submodules.
 */

const config = require('./config');
const logger = require('./logger');
const metrics = require('./metrics');
const tracing = require('./tracing');
const collectors = require('./collectors');
const exporters = require('./exporters');
const health = require('./health');
const middleware = require('./middleware');
const alerts = require('./alerts');
const utils = require('./utils');

module.exports = {
  config,
  logger,
  metrics,
  tracing,
  collectors,
  exporters,
  health,
  middleware,
  alerts,
  utils,
};
