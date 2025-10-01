/**
 * KMS abstraction and simple file-based KMS adapter
 *
 * The KMS exposes:
 * - getCurrentKey() -> { id, keyMaterial (Buffer) }
 * - rotateKey() -> creates a new key and marks as current
 * - getKeyById(id)
 *
 * Adapters can be swapped (AWS KMS, Google KMS) by implementing the same methods.
 *
 * The file-based implementation stores keys encrypted with an env MASTER key if provided,
 * otherwise stores them plaintext (NOT RECOMMENDED in production).
 */

const fs = require('fs');
const path = require('path');
const { hkdf, aesGcmEncrypt, aesGcmDecrypt } = require('./crypto');
const config = require('./config');
const logger = require('./logger');

const MASTER_KEY_ENV = config.CRYPTO.masterKeyEnv;

class FileKMS {
  /**
   * @param {Object} opts
   * @param {string} opts.filePath
   */
  constructor(opts = {}) {
    this.filePath = opts.filePath || config.KMS.fileKeyPath;
    this.store = { keys: [] }; // in-memory cache
    this._load();
  }

  _load() {
    try {
      if (!fs.existsSync(this.filePath)) {
        this.store = { keys: [] };
        this._persist();
        return;
      }
      const raw = fs.readFileSync(this.filePath, 'utf8');
      const parsed = JSON.parse(raw);
      this.store = parsed;
    } catch (err) {
      logger.warn({ err }, 'Failed to load KMS file; creating new store');
      this.store = { keys: [] };
      this._persist();
    }
  }

  _persist() {
    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.store, null, 2), { encoding: 'utf8', mode: 0o600 });
  }

  _encryptKeyMaterial(buf) {
    if (MASTER_KEY_ENV) {
      const master = Buffer.from(MASTER_KEY_ENV, 'base64'); // expecting base64 env var
      const derived = hkdf(master, config.CRYPTO.aesGcmKeyLength, { info: 'kms-file-encrypt' });
      return aesGcmEncrypt({ key: derived, plaintext: buf });
    }
    // no master key: store base64 plaintext (not secure)
    return { iv: null, ciphertext: buf.toString('base64'), tag: null };
  }

  _decryptKeyMaterial(record) {
    if (MASTER_KEY_ENV) {
      const master = Buffer.from(MASTER_KEY_ENV, 'base64');
      const derived = hkdf(master, config.CRYPTO.aesGcmKeyLength, { info: 'kms-file-encrypt' });
      return aesGcmDecrypt({ key: derived, iv: record.iv, ciphertext: record.ciphertext, tag: record.tag });
    }
    return Buffer.from(record.ciphertext, 'base64');
  }

  _createKeyMaterial() {
    // generate AES key bytes
    return cryptoRandomBytes(config.CRYPTO.aesGcmKeyLength);
  }

  get current() {
    return this.store.keys.find((k) => k.current) || null;
  }

  async getCurrentKey() {
    const rec = this.current;
    if (!rec) throw new Error('No current key');
    const material = this._decryptKeyMaterial(rec.material);
    return { id: rec.id, material: material };
  }

  async getKeyById(id) {
    const rec = this.store.keys.find((k) => k.id === id);
    if (!rec) throw new Error('Key not found');
    const material = this._decryptKeyMaterial(rec.material);
    return { id: rec.id, material };
  }

  /**
   * rotateKey: creates a new material and marks it current
   * This function should be atomic; file-based persistence is simple but not atomic across processes.
   */
  async rotateKey(meta = {}) {
    const id = `k-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const materialBuf = await cryptoRandomBytes(config.CRYPTO.aesGcmKeyLength);
    const encrypted = await this._encryptKeyMaterial(materialBuf);
    const rec = {
      id,
      createdAt: new Date().toISOString(),
      material: encrypted,
      meta,
      current: true,
    };
    // clear previous current
    for (const k of this.store.keys) k.current = false;
    this.store.keys.push(rec);
    this._persist();
    logger.info({ id }, 'KMS rotated key');
    return { id, material: materialBuf };
  }
}

/**
 * Helper to get crypto random bytes as Promise
 */
function cryptoRandomBytes(len) {
  return new Promise((resolve, reject) => {
    require('crypto').randomBytes(len, (err, buf) => {
      if (err) return reject(err);
      resolve(buf);
    });
  });
}

// Export default adapter (file-based)
module.exports = {
  create: (opts = {}) => new FileKMS(opts),
};
