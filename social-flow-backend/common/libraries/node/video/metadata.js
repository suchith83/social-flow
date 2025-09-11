/**
 * metadata.js
 *
 * Extract metadata using ffprobe via fluent-ffmpeg.
 * Exposes:
 *  - probe(input) -> returns ffprobe data
 *  - getStreams(input) -> simplified stream info
 *  - getDuration(input)
 */

const ffmpeg = require('fluent-ffmpeg');
const logger = require('./logger');
const config = require('./config');
const { MetadataError } = require('./errors');

if (config.FFPROBE_PATH) ffmpeg.setFfprobePath(config.FFPROBE_PATH);

function probe(input) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(input, (err, data) => {
      if (err) {
        logger.warn({ err }, 'ffprobe failed');
        return reject(new MetadataError('ffprobe failed', { original: err }));
      }
      return resolve(data);
    });
  });
}

async function getStreams(input) {
  const data = await probe(input);
  return (data.streams || []).map((s) => ({
    index: s.index,
    codec: s.codec_name,
    codecLong: s.codec_long_name,
    codecType: s.codec_type,
    width: s.width,
    height: s.height,
    duration: s.duration,
    bit_rate: s.bit_rate,
    sample_rate: s.sample_rate,
    channels: s.channels,
  }));
}

async function getDuration(input) {
  const data = await probe(input);
  return data.format && data.format.duration ? parseFloat(data.format.duration) : null;
}

module.exports = {
  probe,
  getStreams,
  getDuration,
};
