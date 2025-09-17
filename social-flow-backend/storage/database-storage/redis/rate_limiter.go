package redis

import (
	"context"
	"fmt"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// RateLimiter implements a distributed token-bucket limiter using Redis.
// It uses a Lua script for atomic token consumption and refill.
type RateLimiter struct {
	KeyPrefix string
	Script    *redis.Script
	// Config parameters can be per-key; we use tokenBucket script with capacity & refillRate
}

// Lua script:
// KEYS[1] - key
// ARGV[1] - capacity
// ARGV[2] - refill_per_sec (tokens per second as float)
// ARGV[3] - now_ts (milliseconds)
// ARGV[4] - requested_tokens
// Returns: remaining tokens (float) and allowed (1/0)
const tokenBucketLua = `
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local state = redis.call("HMGET", key, "tokens", "last_ts")
local tokens = tonumber(state[1]) or capacity
local last_ts = tonumber(state[2]) or 0

local elapsed = math.max(0, now - last_ts)
local refill_tokens = (elapsed / 1000.0) * refill
tokens = math.min(capacity, tokens + refill_tokens)

local allowed = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
end

redis.call("HMSET", key, "tokens", tokens, "last_ts", now)
-- set a TTL to auto-expire idle buckets (capacity seconds * 2)
redis.call("PEXPIRE", key, math.floor((capacity/refill)*2000))

return {tostring(tokens), allowed}
`

// NewRateLimiter constructs a new limiter.
func NewRateLimiter(prefix string) *RateLimiter {
	return &RateLimiter{
		KeyPrefix: prefix,
		Script:    redis.NewScript(tokenBucketLua),
	}
}

// Allow attempts to consume tokens; returns true if allowed.
func (rl *RateLimiter) Allow(ctx context.Context, key string, capacity float64, refillPerSec float64, requested float64) (bool, float64, error) {
	if Client == nil {
		return false, 0, fmt.Errorf("redis client not initialized")
	}
	fullKey := prefixedKey(rl.KeyPrefix, key)
	now := time.Now().UnixNano() / int64(time.Millisecond)
	res, err := rl.Script.Run(ctx, Client, []string{fullKey}, capacity, refillPerSec, now, requested).Result()
	if err != nil {
		return false, 0, err
	}
	arr, ok := res.([]interface{})
	if !ok || len(arr) < 2 {
		return false, 0, fmt.Errorf("unexpected script response: %#v", res)
	}
	remainingStr := arr[0].(string)
	allowedInt := int64(arr[1].(int64))
	remaining, _ := strconv.ParseFloat(remainingStr, 64)
	return allowedInt == 1, remaining, nil
}
