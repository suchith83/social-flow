/**
 * uploadHandler.js
 *
 * Helpers to integrate with HTTP servers (Express) to accept uploads and kick off processing.
 * - expressUploadEndpoint({ storageAdapter, processor, multiple, maxSize, onComplete })
 *
 * This module does minimal parsing: expects multipart form with 'file' field. For production use,
 * integrate with streaming parsers like busboy or use direct-to-storage presigned uploads.
 *
 * Usage:
 *   const { expressUploadEndpoint } = require('.../video/uploadHandler');
 *   app.post('/upload', expressUploadEndpoint({ storageAdapter, processor, onComplete }));
 */

const busboyFactory = (() => { try { return require('busboy'); } catch (e) { return null; } })();
const path = require('path');
const fs = require('fs');
const os = require('os');
const utils = require('./utils');
const logger = require('./logger');

function expressUploadEndpoint({ storageAdapter, processor, fieldName = 'file', tempBase = undefined, maxFileSize = 500 * 1024 * 1024, onComplete = null } = {}) {
  if (!busboyFactory) throw new Error('busboy is required for expressUploadEndpoint (npm i busboy)');
  return async (req, res, next) => {
    try {
      const bb = busboyFactory({ headers: req.headers, limits: { fileSize: maxFileSize } });
      const files = [];

      const tmpRoot = await utils.createTempDir('video-upload-', tempBase || undefined);

      bb.on('file', (name, file, info) => {
        const { filename, encoding, mimeType } = info;
        if (name !== fieldName) {
          // skip
          file.resume();
          return;
        }
        const saveTo = path.join(tmpRoot, utils.safeFilename(filename || `upload-${utils.randomId(6)}`));
        const ws = fs.createWriteStream(saveTo);
        file.pipe(ws);
        ws.on('finish', () => {
          files.push({ path: saveTo, filename, mimeType, encoding, size: fs.statSync(saveTo).size });
        });
      });

      bb.on('field', (name, val) => {
        // you may capture form fields; not used in this helper
      });

      bb.on('close', async () => {
        if (!files.length) {
          await utils.removeTempDir(tmpRoot);
          return res.status(400).json({ error: 'no files' });
        }
        // For simplicity process first file
        const f = files[0];
        // Optionally upload raw source to storage first
        let sourceRef;
        if (storageAdapter) {
          // upload original to storage location 'uploads/<random>.ext'
          const key = `uploads/${path.basename(f.path)}`;
          await storageAdapter.upload({ key, stream: fs.createReadStream(f.path), size: f.size, contentType: f.mimeType });
          sourceRef = { type: 'storage', key };
        } else {
          sourceRef = { type: 'local', path: f.path };
        }

        // Kick off background processing (don't block request)
        (async () => {
          try {
            const result = await processor.processVideo({ source: sourceRef, storageAdapter, outputPrefix: `videos/${utils.randomId(8)}` });
            logger.info({ result }, 'video processing complete (background)');
            if (onComplete) {
              try { await onComplete(null, result); } catch (e) { logger.warn({ e }, 'onComplete hook failed'); }
            }
          } catch (err) {
            logger.error({ err }, 'background processing failed');
            if (onComplete) {
              try { await onComplete(err); } catch (e) { logger.warn({ e }, 'onComplete hook failed'); }
            }
          } finally {
            // cleanup temp directory
            await utils.removeTempDir(tmpRoot);
          }
        })();

        // Respond with accepted and reference to source
        res.status(202).json({ status: 'accepted', source: sourceRef });
      });

      req.pipe(bb);
    } catch (err) {
      next(err);
    }
  };
}

module.exports = {
  expressUploadEndpoint,
};
