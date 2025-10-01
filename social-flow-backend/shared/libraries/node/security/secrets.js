/**
 * secrets.js
 *
 * Secret storage helpers and envelope encryption using KMS keys.
 *
 * Functions:
 * - encryptSecret(plaintext) -> envelope { keyId, iv, ciphertext, tag }
 * - decryptSecret(envelope) -> plaintext Buffer/string
 *
 * Uses AES-GCM for envelope encryption with per-secret data keys derived from KMS key material.
 */

const { hkdf, aesGcmEncrypt, aesGcmDecrypt } = require('./crypto');
const KMSFactory = require('./kms');
const logger = require('./logger');

const kms = KMSFactory.create(); // default file adapter

/**
 * deriveDataKey: given master key material, derive a per-secret key using HKDF
 */
function deriveDataKey(masterMaterialBuffer, context = '') {
  // derive AES key of configured length
  return hkdf(masterMaterialBuffer, require('./config').CRYPTO.aesGcmKeyLength, { info: `secret:${context}` });
}

/**
 * encryptSecret
 * - plaintext: Buffer or string
 * - context: optional string to bind derivation
 * returns envelope containing keyId and AES-GCM payload
 */
async function encryptSecret(plaintext, context = '') {
  const keyInfo = await kms.getCurrentKey();
  if (!keyInfo) throw new Error('No KMS key available');
  const { id: keyId, material } = keyInfo;
  const dataKey = deriveDataKey(material, context);
  const { iv, ciphertext, tag } = await aesGcmEncrypt({ key: dataKey, plaintext });
  // return envelope (strings)
  return {
    keyId,
    iv,
    ciphertext,
    tag,
    createdAt: new Date().toISOString(),
    context,
  };
}

/**
 * decryptSecret
 * - envelope must contain keyId, iv, ciphertext, tag
 */
async function decryptSecret(envelope) {
  const { keyId, iv, ciphertext, tag } = envelope;
  const keyInfo = await kms.getKeyById(keyId);
  if (!keyInfo) throw new Error('KMS key not found');
  const dataKey = deriveDataKey(keyInfo.material, envelope.context || '');
  const pt = await aesGcmDecrypt({ key: dataKey, iv, ciphertext, tag });
  return pt;
}

module.exports = {
  encryptSecret,
  decryptSecret,
  kmsInstance: kms,
};
