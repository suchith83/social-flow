/**
 * hls.js
 *
 * Helpers to create HLS (HTTP Live Streaming) renditions using ffmpeg.
 * - packageToHls(input, outputDir, renditions, options)
 *   renditions: [{ name, width, bitrate, audioBitrate, maxRate, bufsize }]
 *
 * Produces:
 *  - playlist per rendition (index.m3u8)
 *  - master playlist (master.m3u8)
 *
 * This is a pragmatic implementation: it invokes ffmpeg per rendition, writes segments and playlists.
 * For large-scale production consider using dedicated packager (shaka-packager / ffmpeg streaming with segmenter).
 */

const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');
const utils = require('./utils');
const logger = require('./logger');
const config = require('./config');

if (config.FFMPEG_PATH) ffmpeg.setFfmpegPath(config.FFMPEG_PATH);

/**
 * packageToHls
 * - input: source video path
 * - outputDir: directory to place segments and playlists
 * - renditions: array of { name: '720p', width, height, videoBitrate, audioBitrate }
 */
async function packageToHls(input, outputDir, renditions = [], opts = {}) {
  utils.ensureDirSync(outputDir);
  const segmentDuration = opts.segmentDuration || config.HLS.segmentDuration;
  const masterLines = [
    '#EXTM3U',
    '#EXT-X-VERSION:3',
  ];

  // For simplicity we run ffmpeg per rendition, producing separate playlist files and segments under subdirs
  for (const r of renditions) {
    const renditionDir = path.join(outputDir, r.name);
    utils.ensureDirSync(renditionDir);
    const playlistName = 'index.m3u8';
    const playlistPath = path.join(renditionDir, playlistName);

    // Building args for this rendition
    const args = [
      '-y',
      '-i', input,
      '-map', '0:v:0',
      '-map', '0:a:0',
      '-c:v', r.videoCodec || 'libx264',
      '-b:v', r.videoBitrate || r.bitrate || '1000k',
      '-maxrate', r.maxRate || `${Math.floor(parseInt((r.videoBitrate || '1000k').replace('k','')) * 1.5)}k`,
      '-bufsize', r.bufsize || `${Math.floor(parseInt((r.videoBitrate || '1000k').replace('k','')) * 2)}k`,
      '-vf', `scale=w=${r.width}:h=${r.height}:force_original_aspect_ratio=decrease`,
      '-c:a', r.audioCodec || 'aac',
      '-b:a', r.audioBitrate || '128k',
      '-hls_time', String(segmentDuration),
      '-hls_playlist_type', opts.playlistType || config.HLS.playlistType,
      '-hls_segment_filename', path.join(renditionDir, 'seg-%03d.ts'),
      path.join(renditionDir, playlistName),
    ];

    await new Promise((resolve, reject) => {
      const cmd = ffmpeg()
        .addInput(input)
        .outputOptions(args)
        .on('start', (cmdline) => logger.debug({ cmdline }, 'ffmpeg hls start'))
        .on('error', (err, stdout, stderr) => {
          logger.error({ err, stderr }, 'ffmpeg hls error');
          reject(err);
        })
        .on('end', () => {
          logger.info({ rendition: r.name }, 'hls rendition complete');
          resolve();
        })
        .run();
    });

    // Append to master playlist
    // NOTE: bandwidth field expects bits per second not 'k'
    const bandwidth = Math.floor(parseInt((r.videoBitrate || '1000k').replace('k','')) * 1000);
    masterLines.push(`#EXT-X-STREAM-INF:BANDWIDTH=${bandwidth},RESOLUTION=${r.width}x${r.height}`);
    masterLines.push(`${r.name}/${playlistName}`);
  }

  const masterPath = path.join(outputDir, 'master.m3u8');
  fs.writeFileSync(masterPath, masterLines.join('\n'), 'utf8');

  return { master: masterPath, renditions: renditions.map(r => ({ name: r.name, playlist: path.join(outputDir, r.name, 'index.m3u8') })) };
}

module.exports = {
  packageToHls,
};
