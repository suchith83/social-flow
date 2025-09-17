package redis

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"
)

// Session represents a small session payload stored in Redis.
// For larger session data, store in DB and keep only session ID in cookie.
type Session struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	ExpiresAt time.Time              `json:"expires_at"`
}

// ErrSessionNotFound
var ErrSessionNotFound = errors.New("session not found")

// CreateSession stores a session in Redis and returns session ID. ttl defines session expiration.
func CreateSession(ctx context.Context, s *Session, ttl time.Duration, prefix string) (string, error) {
	if s == nil {
		return "", errors.New("nil session")
	}
	if s.ID == "" {
		s.ID = uuid.New().String()
	}
	now := time.Now().UTC()
	s.CreatedAt = now
	s.ExpiresAt = now.Add(ttl)

	if err := SetJSON(ctx, "session:"+s.ID, s, ttl, prefix); err != nil {
		return "", err
	}
	return s.ID, nil
}

// GetSession fetches session by id.
func GetSession(ctx context.Context, id, prefix string) (*Session, error) {
	var s Session
	if err := GetJSON(ctx, "session:"+id, &s, prefix); err != nil {
		if err == redis.Nil {
			return nil, ErrSessionNotFound
		}
		return nil, err
	}
	// Optionally: refresh TTL on access (sliding sessions)
	return &s, nil
}

// DestroySession deletes a session.
func DestroySession(ctx context.Context, id, prefix string) error {
	return DeleteKey(ctx, "session:"+id, prefix)
}
