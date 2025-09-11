/**
 * crypto.js
 *
 * Cryptography utility functions:
 * - AES-GCM encrypt/decrypt (with authenticated associated data)
 * - RSA key pair generation + sign/verify (PKCS1)
 * - HKDF derive key
 * - HMAC helpers
 *
 * This file intentionally favors safety:
 * - uses Node's crypto.subtle/kdf when available (fallback to crypto)
 * - returns and accepts Buffers for binary clarity
 *
 * NOTE: For truly sensitive operations prefer external KMS (AWS KMS, GCP KMS etc.)
 */

const crypto = require('crypto');
const { promisify } = require('util');
const config = require('./config');
const logger = require('./logger');

const randomBytes = promisify(crypto.randomBytes);

/**
 * AES-GCM encrypt
 * - key: Buffer (length 16/24/32)
 * - plaintext: Buffer | string
 * - aad: Buffer | string | undefined
 * Returns: { iv, ciphertext, tag } as base64 strings (or Buffers if returnBuffers=true)
 */
async function aesGcmEncrypt({ key, plaintext, aad, returnBuffers = false }) {
  if (!Buffer.isBuffer(key)) key = Buffer.from(key, 'utf8');
  const iv = await randomBytes(12); // recommended 12 bytes
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  if (aad) cipher.setAAD(Buffer.isBuffer(aad) ? aad : Buffer.from(aad));
  const ct = Buffer.concat([cipher.update(plaintext, typeof plaintext === 'string' ? 'utf8' : undefined), cipher.final()]);
  const tag = cipher.getAuthTag();
  if (returnBuffers) {
    return { iv, ciphertext: ct, tag };
  }
  return {
    iv: iv.toString('base64'),
    ciphertext: ct.toString('base64'),
    tag: tag.toString('base64'),
  };
}

/**
 * AES-GCM decrypt
 * - key: Buffer
 * - iv/ciphertext/tag: base64 strings or Buffers
 */
async function aesGcmDecrypt({ key, iv, ciphertext, tag }) {
  if (!Buffer.isBuffer(key)) key = Buffer.from(key, 'utf8');
  if (typeof iv === 'string') iv = Buffer.from(iv, 'base64');
  if (typeof ciphertext === 'string') ciphertext = Buffer.from(ciphertext, 'base64');
  if (typeof tag === 'string') tag = Buffer.from(tag, 'base64');
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  if (tag) decipher.setAuthTag(tag);
  const pt = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return pt;
}

/**
 * HKDF (RFC5869) using HMAC-SHA256
 * - Extract & Expand to derive key of length L
 */
function hkdf(ikm, length = 32, { salt = config.CRYPTO.hkdfSalt, info = '' } = {}) {
  // crypto.createHmac for extract
  const hashLen = 32; // SHA-256
  const prk = crypto.createHmac('sha256', salt).update(ikm).digest();
  let t = Buffer.alloc(0);
  let okm = Buffer.alloc(0);
  const n = Math.ceil(length / hashLen);
  for (let i = 0; i < n; i++) {
    const hmac = crypto.createHmac('sha256', prk);
    hmac.update(Buffer.concat([t, Buffer.from(info || ''), Buffer.from([i + 1])]));
    t = hmac.digest();
    okm = Buffer.concat([okm, t]);
  }
  return okm.slice(0, length);
}

/**
 * Generate RSA key pair (2048/3072/4096)
 */
async function generateRsaKeyPair({ modulusLength = 2048 } = {}) {
  const gen = promisify(crypto.generateKeyPair);
  const { publicKey, privateKey } = await gen('rsa', {
    modulusLength,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
  });
  return { publicKey, privateKey };
}

/**
 * RSA sign/verify with SHA256
 */
function rsaSign({ privateKeyPem, message }) {
  const sign = crypto.createSign('sha256');
  sign.update(message);
  sign.end();
  return sign.sign(privateKeyPem, 'base64');
}

function rsaVerify({ publicKeyPem, message, signatureBase64 }) {
  const verify = crypto.createVerify('sha256');
  verify.update(message);
  verify.end();
  return verify.verify(publicKeyPem, signatureBase64, 'base64');
}

/**
 * HMAC helper
 */
function createHmac({ key, data, alg = 'sha256', encoding = 'base64' }) {
  const mac = crypto.createHmac(alg, key).update(data).digest(encoding);
  return mac;
}

/**
 * Constant-time compare
 */
function timingSafeEqual(a, b) {
  try {
    const A = Buffer.isBuffer(a) ? a : Buffer.from(a);
    const B = Buffer.isBuffer(b) ? b : Buffer.from(b);
    if (A.length !== B.length) return false;
    return crypto.timingSafeEqual(A, B);
  } catch (e) {
    return false;
  }
}

module.exports = {
  aesGcmEncrypt,
  aesGcmDecrypt,
  hkdf,
  generateRsaKeyPair,
  rsaSign,
  rsaVerify,
  createHmac,
  timingSafeEqual,
};
