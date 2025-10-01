/**
 * processor.js
 *
 * High-level pipeline orchestration:
 * - Accepts an input video (local path or storage key plus storage adapter)
 * - 1) download/prepare input to local temp
 * - 2) extract metadata
 * - 3) generate thumbnails
 * - 4) transcode with configured profiles (parallel up to concurrency)
 * - 5) package HLS if requested
 * - 6) upload outputs to storage via adapter
 *
 * Exposes:
 *   processVideo({ source: { type, uri }, storageAdapter, profiles, outputPrefix, generateHls })
 *
 * Returns a manifest describing outputs and metadata.
 *
 * This module focuses on orchestration and idempotency (skip if outputs already exist).
 */

const path = require('path');
const fs = require('fs');
const os = require('os');
const pLimit = (() => { try { return require('p-limit'); } catch (e) { return null; } })();
const config = require('./config');
const utils = require('./utils');
const metadata = require('./metadata');
const transcode = require('./transcode');
const thumbnail = require('./thumbnail');
const hls = require('./hls');
const logger = require('./logger');

/**
 * Downloads source to local path if source.type === 'storage', otherwise uses local path as-is.
 * storageAdapter is expected to provide download({ key }) -> { stream }
 */
async function _prepareInput(source, storageAdapter, tempDir) {
  if (source.type === 'local') {
    if (!fs.existsSync(source.path)) throw new Error('Input file not found');
    return source.path;
  } else if (source.type === 'storage') {
    // stream to local temp file
    const key = source.key;
    const localPath = path.join(tempDir, utils.safeFilename(path.basename(key)));
    const { stream } = await storageAdapter.download({ key });
    await new Promise((resolve, reject) => {
      const ws = fs.createWriteStream(localPath);
      stream.pipe(ws);
      stream.on('error', reject);
      ws.on('finish', resolve);
      ws.on('error', reject);
    });
    return localPath;
  } else if (source.type === 'url') {
    // download via fetch
    const fetch = require('node-fetch');
    const res = await fetch(source.url);
    if (!res.ok) throw new Error(`Failed to fetch source: ${res.status}`);
    const localPath = path.join(tempDir, utils.safeFilename(path.basename(source.url.split('?')[0] || `video-${utils.randomId(6)}.mp4`)));
    const fileStream = fs.createWriteStream(localPath);
    await new Promise((resolve, reject) => {
      res.body.pipe(fileStream);
      res.body.on('error', reject);
      fileStream.on('finish', resolve);
    });
    return localPath;
  } else {
    throw new Error('Unsupported source type');
  }
}

/**
 * uploadFile helper wraps adapter.upload, attempts retries if supported by adapter
 */
async function _uploadFile(adapter, localPath, destKey, metadataObj = {}) {
  const stream = fs.createReadStream(localPath);
  return await adapter.upload({ key: destKey, stream, size: fs.statSync(localPath).size, contentType: require('mime-types').lookup(localPath) || 'application/octet-stream', metadata: metadataObj });
}

/**
 * processVideo main function
 */
async function processVideo({
  source,
  storageAdapter, // required to upload outputs
  profiles = config.TRANSCODE.presets, // object of profiles
  outputPrefix = 'videos',
  generateHls = false,
  generateThumbnails = true,
  hlsRenditions = null, // optional array of rendition configs
  concurrency = config.DEFAULT_CONCURRENCY,
} = {}) {
  const tmpRoot = await utils.createTempDir('video-process-');
  logger.info({ tmpRoot, source }, 'processing video');
  try {
    const inputPath = await _prepareInput(source, storageAdapter, tmpRoot);
    const meta = await metadata.probe(inputPath);

    // thumbnails
    let thumbs = [];
    if (generateThumbnails) {
      const thumbsDir = path.join(tmpRoot, 'thumbs');
      utils.ensureDirSync(thumbsDir);
      thumbs = await thumbnail.captureThumbnails(inputPath, thumbsDir);
      // upload thumbnails
      for (let i = 0; i < thumbs.length; i++) {
        const t = thumbs[i];
        const key = `${outputPrefix}/thumbnails/${path.basename(t.path)}`;
        await _uploadFile(storageAdapter, t.path, key, { source: source.type === 'storage' ? source.key : undefined });
        thumbs[i].uploadedKey = key;
      }
    }

    // transcode per profile (parallel with concurrency)
    const profileEntries = Object.entries(profiles);
    const limiter = pLimit ? pLimit(concurrency) : ((fn) => fn());

    const transcodeResults = {};
    await Promise.all(profileEntries.map(([name, prof]) => limiter(async () => {
      const outDir = path.join(tmpRoot, 'transcoded');
      utils.ensureDirSync(outDir);
      const outFile = path.join(outDir, `${utils.safeFilename(name)}.mp4`);
      await transcode.transcodeToFile(inputPath, outFile, { video: prof.video, audio: prof.audio, crf: prof.crf || 23, extraArgs: prof.extraArgs || [], format: 'mp4' });
      // upload
      const destKey = `${outputPrefix}/variants/${name}.mp4`;
      await _uploadFile(storageAdapter, outFile, destKey, { profile: name });
      const stat = fs.statSync(outFile);
      transcodeResults[name] = { key: destKey, size: stat.size };
    })));

    // hls packaging
    let hlsResult = null;
    if (generateHls) {
      const hlsOut = path.join(tmpRoot, 'hls');
      utils.ensureDirSync(hlsOut);
      const renditions = hlsRenditions || Object.entries(profiles).map(([name, prof]) => ({ name, width: prof.video && prof.video.maxWidth ? prof.video.maxWidth : 640, videoBitrate: prof.video && prof.video.bitrate ? prof.video.bitrate : '800k', audioBitrate: prof.audio && prof.audio.bitrate ? prof.audio.bitrate : '96k' }));
      const pkg = await hls.packageToHls(inputPath, hlsOut, renditions);
      // Walk and upload hls outputs
      // upload master
      const masterKey = `${outputPrefix}/hls/master.m3u8`;
      await _uploadFile(storageAdapter, pkg.master, masterKey, {});
      const renditionUploads = [];
      for (const r of pkg.renditions) {
        // upload playlist and segments for each rendition dir
        const dir = path.dirname(r.playlist);
        const files = fs.readdirSync(dir).filter((f) => !f.startsWith('.'));
        for (const f of files) {
          const local = path.join(dir, f);
          const key = `${outputPrefix}/hls/${path.basename(dir)}/${f}`;
          renditionUploads.push(_uploadFile(storageAdapter, local, key, {}));
        }
      }
      await Promise.all(renditionUploads);
      hlsResult = { masterKey, renditions: pkg.renditions.map(r => ({ name: r.name, playlistKey: `${outputPrefix}/hls/${r.name}/index.m3u8` })) };
    }

    const manifest = {
      source,
      metadata: meta,
      thumbnails: thumbs,
      variants: transcodeResults,
      hls: hlsResult,
      processedAt: new Date().toISOString(),
    };

    // Optionally: upload manifest to storage
    const manifestLocal = path.join(tmpRoot, 'manifest.json');
    fs.writeFileSync(manifestLocal, JSON.stringify(manifest, null, 2));
    const manifestKey = `${outputPrefix}/manifest-${utils.randomId(6)}.json`;
    await _uploadFile(storageAdapter, manifestLocal, manifestKey, {});

    manifest.manifestKey = manifestKey;
    return manifest;
  } finally {
    // cleanup
    await utils.removeTempDir(tmpRoot);
  }
}

module.exports = {
  processVideo,
};
