/**
 * s3Adapter.js
 * Adapter for AWS S3 (or S3-compatible endpoints)
 *
 * Features:
 * - streaming upload (putObject for small; multipartUpload for large)
 * - download as stream
 * - stat / headObject
 * - deleteObject
 * - listObjectsV2 iterator
 * - presign GET/PUT
 *
 * Uses AWS SDK v3 (modular). This file tries to avoid pulling heavy deps unless used.
 */

const {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
  DeleteObjectCommand,
  ListObjectsV2Command,
  CreateMultipartUploadCommand,
  UploadPartCommand,
  CompleteMultipartUploadCommand,
  AbortMultipartUploadCommand,
} = (() => {
  try {
    // require lazily; if not installed the import will throw
    const aws = require('@aws-sdk/client-s3');
    return aws;
  } catch (e) {
    // rethrow more helpful error once used
    throw new Error('Please install @aws-sdk/client-s3 to use S3 adapter: npm i @aws-sdk/client-s3');
  }
})();

const { Upload } = (() => {
  try {
    return require('@aws-sdk/lib-storage');
  } catch (e) {
    // lib-storage provides higher-level upload; optional but recommended
    return null;
  }
})() || {};

const { S3RequestPresigner, getSignedUrl } = (() => {
  try {
    const presign = require('@aws-sdk/s3-request-presigner');
    return presign;
  } catch (e) {
    return null;
  }
})() || {};

const Adapter = require('../adapter');
const config = require('../config');
const utils = require('../utils');
const logger = require('../logger');
const fs = require('fs');

class S3Adapter extends Adapter {
  constructor(opts = {}) {
    super(opts);
    this.bucket = opts.bucket || config.S3.bucket;
    const clientOpts = {
      region: opts.region || config.S3.region,
    };
    if (opts.endpoint || config.S3.endpoint) {
      clientOpts.endpoint = opts.endpoint || config.S3.endpoint;
      clientOpts.forcePathStyle = true;
    }
    if (opts.credentials || (config.S3.accessKeyId && config.S3.secretAccessKey)) {
      clientOpts.credentials = opts.credentials || {
        accessKeyId: config.S3.accessKeyId,
        secretAccessKey: config.S3.secretAccessKey,
      };
    }
    this.client = new S3Client(clientOpts);
    this.multipartThreshold = opts.multipartThreshold || config.UPLOAD.multipartThresholdBytes;
    this.partSize = opts.partSize || config.UPLOAD.multipartPartSize;
  }

  async init() {
    if (!this.bucket) throw new Error('S3 bucket not configured');
    logger.info({ bucket: this.bucket }, 'S3Adapter initialized');
  }

  /**
   * upload: streams or buffers using lib-storage upload if available; fallback to putObject.
   */
  async upload({ key, stream, buffer, size, contentType, metadata = {} }) {
    if (!key) throw new Error('upload requires key');
    const params = {
      Bucket: this.bucket,
      Key: key,
      ContentType: contentType || utils.guessMime(key),
      Metadata: metadata,
    };

    try {
      // prefer the high-level Upload from @aws-sdk/lib-storage if available
      if (Upload) {
        const body = buffer || stream;
        const uploader = new Upload({
          client: this.client,
          params: { ...params, Body: body },
          queueSize: 4,
          partSize: this.partSize,
        });
        const res = await uploader.done();
        // res may not be consistent; return key & etag if available
        return { key, size: size || (buffer ? buffer.length : null), etag: res.ETag || null };
      }

      // fallback: for small objects use PutObject
      if (buffer) {
        await this.client.send(new PutObjectCommand({ ...params, Body: buffer }));
        return { key, size: buffer.length, etag: null };
      }
      // if stream provided but lib-storage not available, buffer it (risky for large)
      const bodyBuf = await utils.streamToBuffer(stream);
      await this.client.send(new PutObjectCommand({ ...params, Body: bodyBuf }));
      return { key, size: bodyBuf.length, etag: null };
    } catch (err) {
      logger.error({ err }, 's3 upload error');
      throw err;
    }
  }

  async download({ key }) {
    try {
      const res = await this.client.send(new GetObjectCommand({ Bucket: this.bucket, Key: key }));
      // res.Body is a stream
      return {
        stream: res.Body,
        size: res.ContentLength,
        contentType: res.ContentType,
        metadata: res.Metadata || {},
      };
    } catch (err) {
      if (err.name === 'NoSuchKey' || err.$metadata?.httpStatusCode === 404) {
        throw new Error('Not found');
      }
      throw err;
    }
  }

  async stat({ key }) {
    try {
      const res = await this.client.send(new HeadObjectCommand({ Bucket: this.bucket, Key: key }));
      return {
        key,
        size: res.ContentLength,
        lastModified: res.LastModified,
        etag: res.ETag,
        metadata: res.Metadata || {},
      };
    } catch (err) {
      throw err;
    }
  }

  async delete({ key }) {
    try {
      await this.client.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: key }));
      return { deleted: true };
    } catch (err) {
      throw err;
    }
  }

  async *list(prefix = '', opts = {}) {
    // yields objects in pages (generator)
    let ContinuationToken = undefined;
    do {
      const res = await this.client.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          Prefix: prefix,
          ContinuationToken,
          MaxKeys: opts.maxKeys || 1000,
        })
      );
      for (const obj of res.Contents || []) {
        yield { key: obj.Key, size: obj.Size, lastModified: obj.LastModified, etag: obj.ETag };
      }
      ContinuationToken = res.IsTruncated ? res.NextContinuationToken : undefined;
    } while (ContinuationToken);
  }

  async presignGet(key, { expiresSec = undefined } = {}) {
    if (!getSignedUrl) throw new Error('Install @aws-sdk/s3-request-presigner to use presign');
    const cmd = new GetObjectCommand({ Bucket: this.bucket, Key: key });
    const url = await getSignedUrl(this.client, cmd, { expiresIn: expiresSec || config.PRESIGN.urlExpiresSec });
    return url;
  }

  async presignPut(key, { contentType = 'application/octet-stream', expiresSec = undefined } = {}) {
    if (!getSignedUrl) throw new Error('Install @aws-sdk/s3-request-presigner to use presign');
    const cmd = new PutObjectCommand({ Bucket: this.bucket, Key: key, ContentType: contentType });
    const url = await getSignedUrl(this.client, cmd, { expiresIn: expiresSec || config.PRESIGN.urlExpiresSec });
    return url;
  }
}

module.exports = S3Adapter;
