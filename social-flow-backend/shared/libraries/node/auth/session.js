/**
 * Secure session management using express-session and Redis (if available)
 */

const session = require('express-session');
const RedisStore = require('connect-redis').default;
const { createClient } = require('redis');
const config = require('./config');

let redisClient;
try {
  redisClient = createClient({ legacyMode: true });
  redisClient.connect().catch(console.error);
} catch (err) {
  console.warn('⚠ Redis not available, falling back to in-memory sessions');
}

module.exports = session({
  store: redisClient
    ? new RedisStore({ client: redisClient })
    : undefined,
  secret: config.session.secret,
  resave: false,
  saveUninitialized: false,
  cookie: {
    name: config.session.cookieName,
    maxAge: config.session.maxAge,
    httpOnly: true,
    secure: config.session.secure,
  },
});
