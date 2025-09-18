/**
 * token-manager.js
 *
 * Centralized token acquisition and caching. Scenarios can call functions:
 * - getToken(context, events, next)
 * - refreshTokenIfNeeded(context, events, next)
 *
 * This file demonstrates best practices:
 * - Local in-process cache to avoid re-auth calls on every virtual user iteration
 * - TTL based refresh and locking to prevent thundering herd
 *
 * NOTE: This is memory-cached per Artillery process; in distributed runs you can't rely on this across separate nodes.
 */

const fetch = require('node-fetch');
const { env } = require('../artillery.config');

let cachedToken = null;
let tokenExpiry = 0;
let acquiring = false;

/**
 * Acquire a fresh token from /auth/login and cache it.
 */
async function acquireToken() {
  // simple lock to prevent concurrent acquisitions
  if (acquiring) {
    // wait until available
    await new Promise((resolve) => setTimeout(resolve, 100));
    return cachedToken;
  }
  acquiring = true;
  try {
    const resp = await fetch(`${env.baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: env.username, password: env.password })
    });
    if (!resp.ok) {
      acquiring = false;
      throw new Error(`Auth failed with status ${resp.status}`);
    }
    const body = await resp.json();
    cachedToken = body.token || (body.data && body.data.token) || null;
    // set expiry TTL conservatively (e.g., now + expires_in - 30s)
    const expiresIn = body.expires_in || 3600;
    tokenExpiry = Date.now() + (expiresIn * 1000) - (30 * 1000);
    acquiring = false;
    return cachedToken;
  } catch (err) {
    acquiring = false;
    throw err;
  }
}

module.exports = {
  /**
   * Artillery processor `init` optionally sets up global vars
   */
  init: function (script, util, envVars, next) {
    // no-op for now
    next();
  },

  /**
   * getToken: exporter used directly in YAML as function
   */
  getToken: async function (context, events, next) {
    try {
      if (!cachedToken || Date.now() >= tokenExpiry) {
        await acquireToken();
      }
      context.vars.authToken = cachedToken;
      return next();
    } catch (err) {
      events.emit('counter', 'token_acquire_fail', 1);
      return next(err);
    }
  },

  /**
   * refreshTokenIfNeeded: used as scenario function to refresh token if near expiry
   */
  refreshTokenIfNeeded: async function (context, events, next) {
    try {
      if (!cachedToken || (Date.now() >= (tokenExpiry - 60 * 1000))) {
        await acquireToken();
      }
      context.vars.authToken = cachedToken;
      return next();
    } catch (err) {
      events.emit('counter', 'token_refresh_fail', 1);
      return next(err);
    }
  }
};
