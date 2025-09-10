// ========================================
// File: session.go
// ========================================
package auth

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// SessionStore defines interface for session storage
type SessionStore interface {
	Save(ctx context.Context, sessionID string, data map[string]interface{}, ttl time.Duration) error
	Get(ctx context.Context, sessionID string) (map[string]interface{}, error)
	Destroy(ctx context.Context, sessionID string) error
}

// ================ Redis Session Store ================
type RedisSessionStore struct {
	client *redis.Client
}

func NewRedisSessionStore(addr string, password string, db int) *RedisSessionStore {
	return &RedisSessionStore{
		client: redis.NewClient(&redis.Options{Addr: addr, Password: password, DB: db}),
	}
}

func (r *RedisSessionStore) Save(ctx context.Context, sessionID string, data map[string]interface{}, ttl time.Duration) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return r.client.Set(ctx, sessionID, jsonData, ttl).Err()
}

func (r *RedisSessionStore) Get(ctx context.Context, sessionID string) (map[string]interface{}, error) {
	val, err := r.client.Get(ctx, sessionID).Result()
	if err != nil {
		return nil, err
	}
	var result map[string]interface{}
	err = json.Unmarshal([]byte(val), &result)
	return result, err
}

func (r *RedisSessionStore) Destroy(ctx context.Context, sessionID string) error {
	return r.client.Del(ctx, sessionID).Err()
}

// ================ In-memory Fallback ================
type InMemorySessionStore struct {
	sessions map[string]map[string]interface{}
	expiry   map[string]time.Time
	mu       sync.RWMutex
}

func NewInMemorySessionStore() *InMemorySessionStore {
	return &InMemorySessionStore{
		sessions: make(map[string]map[string]interface{}),
		expiry:   make(map[string]time.Time),
	}
}

func (m *InMemorySessionStore) Save(ctx context.Context, sessionID string, data map[string]interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[sessionID] = data
	m.expiry[sessionID] = time.Now().Add(ttl)
	return nil
}

func (m *InMemorySessionStore) Get(ctx context.Context, sessionID string) (map[string]interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if exp, ok := m.expiry[sessionID]; ok && time.Now().Before(exp) {
		return m.sessions[sessionID], nil
	}
	return nil, errors.New("session expired or not found")
}

func (m *InMemorySessionStore) Destroy(ctx context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, sessionID)
	delete(m.expiry, sessionID)
	return nil
}
