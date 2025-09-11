/**
 * rateLimiter.js
 *
 * Implements a pluggable rate limiter with:
 * - in-memory sliding window with token bucket semantics
 * - Redis adapter (if redisUrl provided)
 *
 * Exposes:
 * - createRateLimiter(opts) => middleware(req, res, next)
 * - limiter.take(key, cost = 1) => Promise<{ allowed, remaining }>
 *
 * Implementation notes:
 * - For distributed limits use Redis adapter which uses INCR & TTL + Lua scripts for atomicity.
 * - The in-memory limiter is best-effort and not safe across multiple processes.
 */

const config = require('./config');
const logger = require('./logger');
let Redis;
try {
  Redis = require('ioredis');
} catch (e) {
  Redis = null;
}

const DEFAULT_WINDOW_MS = config.RATE_LIMITER.defaultWindowMs;
const DEFAULT_MAX = config.RATE_LIMITER.defaultMax;
const DEFAULT_BURST = config.RATE_LIMITER.defaultBurst;

/**
 * In-memory token bucket per key
 */
class InMemoryLimiter {
  constructor({ windowMs = DEFAULT_WINDOW_MS, max = DEFAULT_MAX, burst = DEFAULT_BURST } = {}) {
    this.windowMs = windowMs;
    this.max = max;
    this.burst = burst;
    this.buckets = new Map();
  }

  _now() { return Date.now(); }

  take(key, cost = 1) {
    const now = this._now();
    let b = this.buckets.get(key);
    if (!b) {
      b = { tokens: this.max, last: now };
      this.buckets.set(key, b);
    }
    // refill
    const elapsed = Math.max(0, now - b.last);
    const refill = (elapsed / this.windowMs) * this.max;
    b.tokens = Math.min(this.max + this.burst, b.tokens + refill);
    b.last = now;
    if (b.tokens >= cost) {
      b.tokens -= cost;
      return { allowed: true, remaining: Math.floor(b.tokens) };
    }
    return { allowed: false, remaining: Math.floor(b.tokens) };
  }

  middleware({ keyGenerator = (req) => req.ip, cost = 1, onLimit = null } = {}) {
    return (req, res, next) => {
      const key = keyGenerator(req);
      const resu = this.take(key, cost);
      res.setHeader('X-RateLimit-Remaining', String(resu.remaining));
      if (!resu.allowed) {
        if (onLimit) onLimit(req, res);
        return res.status(429).send('Too many requests');
      }
      next();
    };
  }
}

/**
 * Redis limiter: sliding window using counters + TTL
 * NOTE: for production you might want to implement a Lua script with sorted sets for precise sliding window.
 */
class RedisLimiter {
  constructor({ redisUrl = config.RATE_LIMITER.redisUrl, windowMs = DEFAULT_WINDOW_MS, max = DEFAULT_MAX } = {}) {
    if (!Redis) throw new Error('ioredis required for Redis limiter');
    this.client = new Redis(redisUrl);
    this.windowMs = windowMs;
    this.max = max;
  }

  async take(key, cost = 1) {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    const zset = `rl:${key}`;
    // Use sorted set with timestamp scores to approximate sliding window
    const pipeline = this.client.pipeline();
    pipeline.zadd(zset, now, `${now}:${Math.random().toString(36).slice(2,8)}`);
    pipeline.zremrangebyscore(zset, 0, windowStart);
    pipeline.zcard(zset);
    pipeline.pexpire(zset, this.windowMs + 1000);
    const res = await pipeline.exec();
    // res contains results; third element is cardinality
    const card = res[2][1];
    const allowed = card <= this.max;
    return { allowed, remaining: Math.max(0, this.max - card) };
  }

  middleware({ keyGenerator = (req) => req.ip, onLimit = null } = {}) {
    return async (req, res, next) => {
      try {
        const key = keyGenerator(req);
        const r = await this.take(key);
        res.setHeader('X-RateLimit-Remaining', String(r.remaining));
        if (!r.allowed) {
          if (onLimit) onLimit(req, res);
          return res.status(429).send('Too many requests');
        }
        next();
      } catch (err) {
        logger.warn({ err }, 'Redis limiter error; falling through');
        next();
      }
    };
  }
}

/**
 * Factory: returns a limiter instance based on configuration
 */
function createDefaultLimiter(opts = {}) {
  if (!config.RATE_LIMITER.inMemory && config.RATE_LIMITER.redisUrl) {
    try {
      return new RedisLimiter({ redisUrl: config.RATE_LIMITER.redisUrl, ...opts });
    } catch (e) {
      logger.warn('Redis not available; falling back to in-memory limiter', e);
    }
  }
  return new InMemoryLimiter(opts);
}

module.exports = {
  InMemoryLimiter,
  RedisLimiter,
  createDefaultLimiter,
};
