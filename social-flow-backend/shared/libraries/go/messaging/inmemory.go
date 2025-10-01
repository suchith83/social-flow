// ========================================
// File: inmemory.go
// ========================================
package messaging

import (
	"context"
	"sync"
)

type InMemory struct {
	subscribers map[string][]func(Message) error
	mu          sync.RWMutex
}

func NewInMemory() *InMemory {
	return &InMemory{subscribers: make(map[string][]func(Message) error)}
}

func (m *InMemory) Publish(ctx context.Context, msg Message) error {
	m.mu.RLock()
	handlers := m.subscribers[msg.Topic]
	m.mu.RUnlock()
	for _, h := range handlers {
		_ = h(msg)
	}
	return nil
}

func (m *InMemory) Subscribe(ctx context.Context, topic string, handler func(Message) error) error {
	m.mu.Lock()
	m.subscribers[topic] = append(m.subscribers[topic], handler)
	m.mu.Unlock()
	return nil
}

func (m *InMemory) Close() error { return nil }
