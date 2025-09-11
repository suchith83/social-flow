/**
 * Centralized configuration loader for the auth library
 * Uses environment variables with safe defaults
 */

require('dotenv').config();

module.exports = {
  jwt: {
    secret: process.env.JWT_SECRET || 'super-secure-secret',
    expiresIn: process.env.JWT_EXPIRES_IN || '1h',
    issuer: process.env.JWT_ISSUER || 'social-flow',
  },
  oauth: {
    googleClientId: process.env.GOOGLE_CLIENT_ID || '',
    googleClientSecret: process.env.GOOGLE_CLIENT_SECRET || '',
    githubClientId: process.env.GITHUB_CLIENT_ID || '',
    githubClientSecret: process.env.GITHUB_CLIENT_SECRET || '',
    redirectUri: process.env.OAUTH_REDIRECT_URI || 'http://localhost:3000/callback',
  },
  session: {
    cookieName: process.env.SESSION_COOKIE_NAME || 'sid',
    secret: process.env.SESSION_SECRET || 'session-secret',
    maxAge: parseInt(process.env.SESSION_MAX_AGE || '3600000', 10), // 1 hour
    secure: process.env.NODE_ENV === 'production',
  },
  security: {
    saltRounds: parseInt(process.env.BCRYPT_SALT_ROUNDS || '12', 10),
    mfaWindow: parseInt(process.env.MFA_WINDOW || '1', 10), // TOTP allowed drift
  },
  audit: {
    logFile: process.env.AUDIT_LOG_FILE || './logs/audit.log',
  },
};
