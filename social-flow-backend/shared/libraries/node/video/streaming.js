/**
 * streaming.js
 *
 * Utilities for serving media files over HTTP, supporting Range requests
 * - serveFile(req, res, filePath, contentType)
 *
 * This helper is minimal but covers common behavior: 206 Partial Content, Accept-Ranges header, content-type, caching.
 */

const fs = require('fs');
const path = require('path');
const logger = require('./logger');

function serveFile(req, res, filePath, contentType = 'application/octet-stream', opts = {}) {
  // opts:{ cacheControl }
  fs.stat(filePath, (err, stat) => {
    if (err) {
      res.statusCode = 404;
      res.end('Not found');
      return;
    }

    const total = stat.size;
    const range = req.headers.range;
    res.setHeader('Accept-Ranges', 'bytes');

    if (opts.cacheControl) res.setHeader('Cache-Control', opts.cacheControl);
    if (!range) {
      res.writeHead(200, { 'Content-Type': contentType, 'Content-Length': total });
      fs.createReadStream(filePath).pipe(res);
      return;
    }

    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : total - 1;
    if (Number.isNaN(start) || Number.isNaN(end) || start > end || end >= total) {
      res.writeHead(416, { 'Content-Range': `bytes */${total}` });
      res.end();
      return;
    }
    const chunkSize = (end - start) + 1;
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': contentType,
    });
    const stream = fs.createReadStream(filePath, { start, end });
    stream.pipe(res);
  });
}

module.exports = {
  serveFile,
};
