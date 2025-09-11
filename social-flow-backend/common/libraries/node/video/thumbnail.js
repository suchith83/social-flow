/**
 * thumbnail.js
 *
 * Generate thumbnails using ffmpeg (frame extraction) + sharp (resize/quality).
 * Exposes:
 *  - captureThumbnails(inputPath, outputDir, options) -> array of file paths
 *  - captureSingle(input, outputPath, timeOffsetSec)
 *
 * Approach:
 *  - use ffmpeg to seek to approximate timestamps and write single-frame images to disk
 *  - post-process with sharp for quality/size
 *
 * Note: extracting frames from a stream can be tricky; prefer file inputs or presigned local files.
 */

const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');
const sharp = (() => { try { return require('sharp'); } catch (e) { return null; } })();
const logger = require('./logger');
const config = require('./config');
const utils = require('./utils');
const { ThumbnailError } = require('./errors');

if (config.FFMPEG_PATH) ffmpeg.setFfmpegPath(config.FFMPEG_PATH);
if (config.FFPROBE_PATH) ffmpeg.setFfprobePath(config.FFPROBE_PATH);

/**
 * getDuration: helper to get duration via ffprobe (delegates to metadata module if exists)
 */
const metadataLib = require('./metadata');

/**
 * captureSingle: seek to time (sec) and save single frame to outputPath
 */
function captureSingle(input, outputPath, timeSec = 1) {
  return new Promise((resolve, reject) => {
    const cmd = ffmpeg(input)
      .on('start', (cmdline) => logger.debug({ cmdline }, 'ffmpeg capture start'))
      .on('error', (err) => {
        logger.warn({ err }, 'ffmpeg capture error');
        reject(new ThumbnailError('ffmpeg capture failed', { original: err }));
      })
      .on('end', () => {
        resolve(outputPath);
      })
      // seek then output single frame
      .screenshots({
        timestamps: [timeSec],
        filename: path.basename(outputPath),
        folder: path.dirname(outputPath),
        size: '1920x?',
      });
    // Note: fluent-ffmpeg's screenshots creates the file and ends the command.
  });
}

/**
 * processWithSharp: resize/quality adjust if sharp available
 */
async function processWithSharp(inputPath, outputPath, { width = config.THUMBNAIL.width, height = config.THUMBNAIL.height, quality = config.THUMBNAIL.quality } = {}) {
  if (!sharp) {
    // fallback: leave as extracted
    return outputPath;
  }
  await sharp(inputPath).resize(width, height, { fit: 'cover' }).jpeg({ quality }).toFile(outputPath);
  return outputPath;
}

/**
 * captureThumbnails:
 * - compute duration, pick N timestamps (avoid first/last few seconds)
 * - capture frames and post-process with sharp
 */
async function captureThumbnails(input, outputDir, options = {}) {
  const count = options.count || config.THUMBNAIL.count;
  const width = options.width || config.THUMBNAIL.width;
  const height = options.height || config.THUMBNAIL.height;
  const quality = options.quality || config.THUMBNAIL.quality;

  utils.ensureDirSync(outputDir);

  const meta = await metadataLib.probe(input).catch((e) => {
    logger.warn({ err: e }, 'failed to probe for thumbnail duration; defaulting timestamps');
    return null;
  });

  const duration = meta && meta.format && meta.format.duration ? parseFloat(meta.format.duration) : null;
  const timestamps = [];
  if (duration && duration > 5) {
    // pick evenly spread times avoiding first/last 1 second
    const start = Math.min(1, duration * 0.05);
    const end = Math.max(duration - 1, duration * 0.95);
    for (let i = 0; i < count; i++) {
      timestamps.push(start + ((end - start) * i) / Math.max(1, count - 1));
    }
  } else {
    // fallback: 1s, 2s...
    for (let i = 0; i < count; i++) timestamps.push(1 + i * 2);
  }

  const generated = [];
  for (let i = 0; i < timestamps.length; i++) {
    const t = timestamps[i];
    const tmpFile = path.join(outputDir, `thumb-${i}-${utils.randomId(6)}.jpg`);
    try {
      // capture
      await captureSingle(input, tmpFile, t);
      // post-process
      const outFile = path.join(outputDir, `thumb-${i}.jpg`);
      await processWithSharp(tmpFile, outFile, { width, height, quality });
      // cleanup tmp if different
      if (tmpFile !== outFile && fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
      generated.push({ path: outFile, time: t });
    } catch (err) {
      logger.warn({ err, t }, 'Failed to create thumbnail for timestamp');
    }
  }
  if (!generated.length) throw new ThumbnailError('No thumbnails generated');
  return generated;
}

module.exports = {
  captureSingle,
  captureThumbnails,
};
