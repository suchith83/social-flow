/**
 * Entry point for the Node Auth library
 * Exports all authentication modules in a single unified API.
 */

const jwt = require('./jwt');
const oauth = require('./oauth');
const password = require('./password');
const mfa = require('./mfa');
const session = require('./session');
const rbac = require('./rbac');
const audit = require('./audit');
const utils = require('./utils');
const config = require('./config');

module.exports = {
  jwt,
  oauth,
  password,
  mfa,
  session,
  rbac,
  audit,
  utils,
  config,
};
