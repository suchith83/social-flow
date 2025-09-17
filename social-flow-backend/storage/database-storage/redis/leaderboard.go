package redis

import (
	"context"
	"fmt"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// Leaderboard wraps operations on a Redis Sorted Set used as a leaderboard.
type Leaderboard struct {
	Key    string
	Prefix string
	TTL    time.Duration // optional TTL for leaderboard
}

// NewLeaderboard returns a leaderboard instance.
func NewLeaderboard(key, prefix string, ttl time.Duration) *Leaderboard {
	return &Leaderboard{Key: key, Prefix: prefix, TTL: ttl}
}

// AddOrIncrement increments the score of member by delta (useful for view-counts).
func (l *Leaderboard) AddOrIncrement(ctx context.Context, member string, delta float64) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	k := prefixedKey(l.Prefix, l.Key)
	if err := Client.ZIncrBy(ctx, k, delta, member).Err(); err != nil {
		return err
	}
	if l.TTL > 0 {
		_ = Client.Expire(ctx, k, l.TTL).Err()
	}
	return nil
}

// TopN returns the top N members with their scores (descending).
func (l *Leaderboard) TopN(ctx context.Context, n int64) ([]redis.Z, error) {
	k := prefixedKey(l.Prefix, l.Key)
	items, err := Client.ZRevRangeWithScores(ctx, k, 0, n-1).Result()
	if err != nil {
		return nil, err
	}
	return items, nil
}

// Rank returns 0-based rank (nil if not found). Note: ZRevRank gives rank in descending order.
func (l *Leaderboard) Rank(ctx context.Context, member string) (int64, error) {
	k := prefixedKey(l.Prefix, l.Key)
	rank, err := Client.ZRevRank(ctx, k, member).Result()
	if err == redis.Nil {
		return -1, fmt.Errorf("member not found")
	}
	return rank, err
}

// GetScore returns the score of a member.
func (l *Leaderboard) GetScore(ctx context.Context, member string) (float64, error) {
	k := prefixedKey(l.Prefix, l.Key)
	score, err := Client.ZScore(ctx, k, member).Result()
	if err == redis.Nil {
		return 0, fmt.Errorf("member not found")
	}
	return score, err
}
