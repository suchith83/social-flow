/**
 * Entry point for storage library
 * Exports adapters, client, utilities, and helpers.
 */

const config = require('./config');
const logger = require('./logger');
const errors = require('./errors');
const utils = require('./utils');
const Adapter = require('./adapter');
const LocalFsAdapter = require('./adapters/localFsAdapter');
const S3Adapter = require('./adapters/s3Adapter');
const Client = require('./client');
const Uploader = require('./uploader');
const Presign = require('./presign');
const Lifecycle = require('./lifecycle');
const Multipart = require('./multipart');

module.exports = {
  config,
  logger,
  errors,
  utils,
  Adapter,
  LocalFsAdapter,
  S3Adapter,
  Client,
  Uploader,
  Presign,
  Lifecycle,
  Multipart,
};
