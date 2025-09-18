/**
 * fixtures.js
 *
 * Shared Artillery JS helper functions:
 * - populates dynamic variables (timestamp, random IDs)
 * - prepares payloads and sample data references
 * - used via "processor" in YAML config to expose functions to scenarios
 *
 * Artillery exposes `events` and context; function signatures follow their docs.
 */

const fs = require('fs');
const path = require('path');
const { env } = require('../artillery.config');

// small RNG helper
function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

module.exports = {
  /**
   * Called before any scenario starts when processor is loaded.
   * We prepare some global fixtures (if required).
   */
  init: function (script, util, envVars, next) {
    // load a small fixture file if present
    const samplePath = env.storageSampleFile;
    let sampleExists = false;
    try {
      sampleExists = fs.existsSync(samplePath);
    } catch (err) {
      sampleExists = false;
    }
    script.vars = script.vars || {};
    script.vars.sampleExists = sampleExists;
    next(); // continue
  },

  /**
   * beforeRequest hook to inject timestamp and randomness into each request
   */
  beforeRequest: function (requestParams, context, ee, next) {
    // add per-iteration timestamp
    context.vars.timestamp = new Date().toISOString();
    // generate a pseudo-random item id (used in baseline)
    context.vars.randomItemId = `itm-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    // supply credentials if needed
    context.vars.username = env.username;
    context.vars.password = env.password;
    next();
  },

  /**
   * prepareMultipartUpload:
   * - This function demonstrates how to request a presigned URL/form for multipart upload,
   *   then stores `uploadKey` or `uploadUrl` in context for the next step to use.
   *   (actual uploading is handled in multipartUploadFile using node-fetch/form-data)
   */
  prepareMultipartUpload: async function (context, events, next) {
    try {
      // call the app endpoint to create an upload entry
      // NOTE: Artillery's `context.request` is not available in processors; you can call external HTTP libs.
      // We will store a key for the scenario to use as a reference.
      const key = `perf-upload-${Date.now()}-${Math.floor(Math.random() * 99999)}`;
      context.vars.uploadKey = key;
      return next();
    } catch (err) {
      // mark as failure for the scenario
      events.emit('counter', 'upload_prepare_fail', 1);
      return next(err);
    }
  },

  /**
   * multipartUploadFile:
   * Demonstrates a Node.js-based multipart file upload to a presigned endpoint.
   * This uses `node-fetch` and `form-data`. Note: Artillery's runtime may not include node-fetch;
   * You can implement uploads using the built-in `http`/`https` modules or ensure devDeps include node-fetch.
   *
   * For performance runs, avoid large files, use small sample file or multiple small parts.
   */
  multipartUploadFile: async function (context, events, next) {
    const fetch = require('node-fetch'); // ensure installed if used; otherwise swap to http.request
    const FormData = require('form-data');

    const samplePath = env.storageSampleFile;
    if (!fs.existsSync(samplePath)) {
      events.emit('counter', 'upload_file_missing', 1);
      return next(new Error('sample file missing'));
    }

    // Example: assume our app exposes a presigned POST form for uploads at /storage/presigned-form
    try {
      // Step 1: request presigned form fields
      const presignedResp = await fetch(`${env.baseUrl}/storage/presigned-form`, {
        method: 'POST',
        body: JSON.stringify({ key: context.vars.uploadKey, bucket: env.storageBucket }),
        headers: { 'Content-Type': 'application/json' }
      });
      if (!presignedResp.ok) {
        events.emit('counter', 'presigned_form_fail', 1);
        return next(new Error('presigned form request failed'));
      }
      const presigned = await presignedResp.json();
      // presigned expected to contain: url and fields
      const form = new FormData();
      Object.entries(presigned.fields || {}).forEach(([k, v]) => form.append(k, v));
      form.append('file', fs.createReadStream(samplePath));

      // Step 2: upload file directly to storage provider (S3/Azure/GCS) using presigned form
      const uploadResp = await fetch(presigned.url, { method: 'POST', body: form });
      if (!uploadResp.ok) {
        events.emit('counter', 'upload_fail', 1);
        return next(new Error('file upload failed'));
      }
      events.emit('counter', 'upload_success', 1);
      return next();
    } catch (err) {
      events.emit('counter', 'upload_error', 1);
      return next(err);
    }
  }
};
