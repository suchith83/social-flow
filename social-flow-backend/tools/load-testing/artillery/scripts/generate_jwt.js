const fs = require('fs');
const jwt = require('jsonwebtoken');
const path = require('path');

let PRIVATE_KEY = null;
const keyPath = process.env.SERVICE_ACCOUNT_PRIVATE_KEY_PATH || './keys/service_account.pem';
if (fs.existsSync(keyPath)) {
  PRIVATE_KEY = fs.readFileSync(keyPath, 'utf8');
}

module.exports = {
  setupAuth: function (userContext, events, done) {
    const vu = userContext.vars._vu || (userContext.vars._vu = Math.floor(Math.random() * 1000000));
    const userId = `lt-user-${vu}`;
    userContext.vars.user_id = userId;

    try {
      if (PRIVATE_KEY) {
        const payload = {
          sub: userId,
          iss: process.env.SERVICE_ACCOUNT_ID || 'service-loadtest',
          aud: process.env.ARTILLERY_TARGET || 'service',
          iat: Math.floor(Date.now() / 1000),
          exp: Math.floor(Date.now() / 1000) + 3600
        };
        userContext.vars.auth_token = jwt.sign(payload, PRIVATE_KEY, { algorithm: 'RS256' });
      } else {
        userContext.vars.auth_token = jwt.sign({ sub: userId }, 'dev-secret', { algorithm: 'HS256', expiresIn: '1h' });
      }
    } catch {
      userContext.vars.auth_token = `dummy-token-${userId}`;
    }

    userContext.vars.refresh_token = `refresh-${userId}-${Date.now()}`;
    userContext.vars.headers = userContext.vars.headers || {};
    userContext.vars.headers.Authorization = `Bearer ${userContext.vars.auth_token}`;

    return done();
  },

  $randomInt: function (low, high) {
    return Math.floor(Math.random() * (high - low + 1)) + low;
  }
};
