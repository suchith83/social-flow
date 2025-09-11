/**
 * Entry point for the security library
 * Exposes all main modules with a stable API.
 */

const config = require('./config');
const logger = require('./logger');
const crypto = require('./crypto');
const KMS = require('./kms');
const secrets = require('./secrets');
const helmetWrapper = require('./helmetWrapper');
const csp = require('./csp');
const csrf = require('./csrf');
const rateLimiter = require('./rateLimiter');
const sanitizer = require('./sanitizer');
const validation = require('./validation');
const audit = require('./audit');
const abac = require('./abac');

module.exports = {
  config,
  logger,
  crypto,
  KMS,
  secrets,
  helmetWrapper,
  csp,
  csrf,
  rateLimiter,
  sanitizer,
  validation,
  audit,
  abac,
};
