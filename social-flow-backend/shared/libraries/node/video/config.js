/**
 * configuration for video library
 * pick environment variables or sensible defaults
 */

require('dotenv').config();

const int = (v, d) => (v === undefined ? d : parseInt(v, 10));
const bool = (v, d = false) => (v === undefined ? d : String(v).toLowerCase() === 'true');

module.exports = {
  // ffmpeg/ffprobe paths (if not on PATH set here)
  FFMPEG_PATH: process.env.FFMPEG_PATH || undefined,
  FFPROBE_PATH: process.env.FFPROBE_PATH || undefined,

  // Transcode defaults
  TRANSCODE: {
    presets: {
      // name: { video: { codec, bitrate, maxWidth }, audio: { codec, bitrate }, crf, additionalArgs }
      '1080p': { video: { codec: 'libx264', bitrate: '4000k', maxWidth: 1920 }, audio: { codec: 'aac', bitrate: '128k' }, crf: 23 },
      '720p': { video: { codec: 'libx264', bitrate: '2500k', maxWidth: 1280 }, audio: { codec: 'aac', bitrate: '128k' }, crf: 23 },
      '480p': { video: { codec: 'libx264', bitrate: '1000k', maxWidth: 854 }, audio: { codec: 'aac', bitrate: '96k' }, crf: 24 },
      '360p': { video: { codec: 'libx264', bitrate: '600k', maxWidth: 640 }, audio: { codec: 'aac', bitrate: '64k' }, crf: 26 },
    },
    maxRetries: int(process.env.VIDEO_TRANSCODE_MAX_RETRIES || '3', 3),
    baseBackoffMs: int(process.env.VIDEO_TRANSCODE_BASE_BACKOFF_MS || '300', 300),
  },

  // Thumbnail defaults
  THUMBNAIL: {
    count: int(process.env.VIDEO_THUMBNAIL_COUNT || '3', 3),
    width: int(process.env.VIDEO_THUMBNAIL_WIDTH || '320', 320),
    height: int(process.env.VIDEO_THUMBNAIL_HEIGHT || '180', 180),
    quality: int(process.env.VIDEO_THUMBNAIL_QUALITY || '80', 80),
  },

  // HLS defaults
  HLS: {
    segmentDuration: int(process.env.VIDEO_HLS_SEGMENT_DURATION || '6', 6),
    playlistType: process.env.VIDEO_HLS_PLAYLIST_TYPE || 'vod', // vod or event
  },

  // File system / temp
  TEMP_DIR: process.env.VIDEO_TEMP_DIR || '/tmp/video-lib',
  CLEANUP_TEMP: bool(process.env.VIDEO_CLEANUP_TEMP || true),

  // Concurrency
  DEFAULT_CONCURRENCY: int(process.env.VIDEO_DEFAULT_CONCURRENCY || '2', 2),

  // Logging
  LOG_LEVEL: process.env.VIDEO_LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug'),
};
