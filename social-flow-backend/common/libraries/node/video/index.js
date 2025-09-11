/**
 * common/libraries/node/video/index.js
 * Public entrypoint for the video library.
 */

const config = require('./config');
const logger = require('./logger');
const errors = require('./errors');
const utils = require('./utils');
const transcode = require('./transcode');
const thumbnail = require('./thumbnail');
const metadata = require('./metadata');
const hls = require('./hls');
const streaming = require('./streaming');
const processor = require('./processor');
const uploadHandler = require('./uploadHandler');

module.exports = {
  config,
  logger,
  errors,
  utils,
  transcode,
  thumbnail,
  metadata,
  hls,
  streaming,
  processor,
  uploadHandler,
};
