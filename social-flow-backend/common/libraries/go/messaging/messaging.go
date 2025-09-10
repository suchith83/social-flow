// ========================================
// File: messaging.go
// ========================================
package messaging

import (
	"context"
	"errors"
	"time"
)

// Message represents a generic event/message
type Message struct {
	Topic     string
	Key       string
	Value     []byte
	Timestamp time.Time
	Headers   map[string]string
}

// Publisher publishes messages
type Publisher interface {
	Publish(ctx context.Context, msg Message) error
	Close() error
}

// Subscriber consumes messages
type Subscriber interface {
	Subscribe(ctx context.Context, topic string, handler func(Message) error) error
	Close() error
}

// MessagingManager is the unified abstraction
type MessagingManager struct {
	Publisher
	Subscriber
}

// NewMessagingManager wires publisher + subscriber
func NewMessagingManager(pub Publisher, sub Subscriber) (*MessagingManager, error) {
	if pub == nil || sub == nil {
		return nil, errors.New("publisher and subscriber required")
	}
	return &MessagingManager{Publisher: pub, Subscriber: sub}, nil
}
