/**
 * Multi-Factor Authentication (TOTP based using speakeasy)
 */

const speakeasy = require('speakeasy');
const qrcode = require('qrcode');
const config = require('./config');

module.exports = {
  generateSecret: (username) =>
    speakeasy.generateSecret({
      length: 20,
      name: `SocialFlow (${username})`,
    }),

  generateQRCode: async (otpauthUrl) => {
    return await qrcode.toDataURL(otpauthUrl);
  },

  verifyToken: (secret, token) => {
    return speakeasy.totp.verify({
      secret,
      encoding: 'base32',
      token,
      window: config.security.mfaWindow,
    });
  },
};
