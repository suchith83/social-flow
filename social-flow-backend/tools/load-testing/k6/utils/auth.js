import jwt from 'jsonwebtoken';
import fs from 'fs';

let privateKey = null;
if (process.env.SERVICE_ACCOUNT_PRIVATE_KEY_PATH && fs.existsSync(process.env.SERVICE_ACCOUNT_PRIVATE_KEY_PATH)) {
  privateKey = fs.readFileSync(process.env.SERVICE_ACCOUNT_PRIVATE_KEY_PATH, 'utf8');
}

export function getAuthHeaders() {
  let token;
  try {
    const payload = {
      sub: `lt-user-${Math.floor(Math.random() * 1000000)}`,
      iss: process.env.SERVICE_ACCOUNT_ID || 'service-loadtest',
      aud: process.env.K6_TARGET,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 3600
    };
    if (privateKey) {
      token = jwt.sign(payload, privateKey, { algorithm: 'RS256' });
    } else {
      token = jwt.sign(payload, 'dev-secret', { algorithm: 'HS256' });
    }
  } catch {
    token = `dummy-${Date.now()}`;
  }
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}
