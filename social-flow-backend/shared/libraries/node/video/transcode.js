/**
 * transcode.js
 *
 * High-level transcoding API using fluent-ffmpeg. Exposes functions:
 * - transcodeToFile(inputPath, outputPath, options)
 * - transcodeWithProfiles(inputPath, outputDir, profiles, opts) -> returns mapping of profile->result
 *
 * Features:
 * - Retry with backoff for transient ffmpeg failures
 * - Streams support (input may be a file or readable stream)
 * - Progress events via EventEmitter (if caller wants)
 *
 * NOTE: fluent-ffmpeg spawns ffmpeg and expects ffmpeg on PATH or configured path.
 */

const ffmpeg = require('fluent-ffmpeg');
const { PassThrough } = require('stream');
const EventEmitter = require('events');
const path = require('path');
const fs = require('fs');
const logger = require('./logger');
const config = require('./config');
const utils = require('./utils');
const { TranscodeError } = require('./errors');

if (config.FFMPEG_PATH) ffmpeg.setFfmpegPath(config.FFMPEG_PATH);
if (config.FFPROBE_PATH) ffmpeg.setFfprobePath(config.FFPROBE_PATH);

/**
 * Build ffmpeg command with options
 * options: { preset: {video/audio/crf}, format, extraArgs: [], size: {maxWidth}, hwaccel }
 */
function _buildCommand(input, outputPath, options = {}) {
  const cmd = ffmpeg(input);

  // video codec & options
  if (options.video) {
    cmd.videoCodec(options.video.codec || 'libx264');
    if (options.video.bitrate) cmd.videoBitrate(options.video.bitrate, true);
    if (options.crf !== undefined) cmd.outputOptions(['-crf', String(options.crf)]);
    // scaling
    if (options.video.maxWidth) {
      cmd.videoFilters(`scale='min(${options.video.maxWidth},iw)':-2`);
    }
  } else {
    // copy video stream if nothing specified
    cmd.videoCodec('copy');
  }

  // audio codec
  if (options.audio) {
    if (options.audio.codec) cmd.audioCodec(options.audio.codec);
    if (options.audio.bitrate) cmd.audioBitrate(options.audio.bitrate);
  } else {
    cmd.audioCodec('aac');
  }

  // format / container
  if (options.format) cmd.format(options.format);

  if (Array.isArray(options.extraArgs)) cmd.outputOptions(options.extraArgs);

  // output file
  cmd.output(outputPath);

  return cmd;
}

/**
 * transcodeToFile
 * - input: path or stream
 * - outputPath: destination path
 * - options: see above
 * returns: Promise resolving when transcode done
 */
async function transcodeToFile(input, outputPath, options = {}, { retries = config.TRANSCODE.maxRetries } = {}) {
  const emitter = new EventEmitter();
  let attempt = 0;
  let lastErr = null;

  // helper to run ffmpeg once
  async function runOnce() {
    return new Promise((resolve, reject) => {
      const cmd = _buildCommand(input, outputPath, options);

      cmd.on('start', (cmdLine) => {
        logger.debug({ cmdLine, output: outputPath }, 'ffmpeg start');
        emitter.emit('start', { cmdLine });
      });

      cmd.on('progress', (progress) => {
        emitter.emit('progress', progress);
      });

      cmd.on('error', (err, stdout, stderr) => {
        logger.warn({ errMsg: err.message, stdout: stdout && stdout.toString(), stderr: stderr && stderr.toString(), output: outputPath }, 'ffmpeg error');
        reject(new TranscodeError('ffmpeg failed', { original: err, stdout, stderr }));
      });

      cmd.on('end', () => {
        logger.info({ output: outputPath }, 'ffmpeg completed');
        resolve();
      });

      // run
      try {
        cmd.run();
      } catch (err) {
        reject(err);
      }
    });
  }

  while (attempt <= retries) {
    try {
      await runOnce();
      return { output: outputPath };
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        const wait = utils.exponentialBackoff(attempt);
        logger.warn({ attempt, wait, err: err.message }, 'transcode failed; retrying after backoff');
        await new Promise((res) => setTimeout(res, wait));
        attempt++;
        continue;
      }
      break;
    }
  }
  throw new TranscodeError('Transcode failed after retries', { original: lastErr });
}

/**
 * transcodeWithProfiles
 * - takes an input and a set of named profiles (e.g. presets in config)
 * - produces one file per profile in outputDir, returns map of { profileName: { path, size } }
 */
async function transcodeWithProfiles(input, outputDir, profiles = {}, opts = {}) {
  utils.ensureDirSync(outputDir);
  const results = {};
  for (const [name, profile] of Object.entries(profiles)) {
    const outfile = path.join(outputDir, `${utils.safeFilename(name)}.mp4`);
    const options = { video: profile.video, audio: profile.audio, crf: profile.crf, extraArgs: profile.extraArgs || [], format: 'mp4' };
    await transcodeToFile(input, outfile, options, opts);
    const stat = fs.statSync(outfile);
    results[name] = { path: outfile, size: stat.size };
  }
  return results;
}

module.exports = {
  transcodeToFile,
  transcodeWithProfiles,
};
