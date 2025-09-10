// ========================================
// File: middleware.go
// ========================================
package messaging

import (
	"context"
	"log"
	"time"
)

// LoggingMiddleware wraps publisher/subscriber
type LoggingMiddleware struct {
	Publisher
	Subscriber
}

func NewLoggingMiddleware(pub Publisher, sub Subscriber) *LoggingMiddleware {
	return &LoggingMiddleware{Publisher: pub, Subscriber: sub}
}

func (m *LoggingMiddleware) Publish(ctx context.Context, msg Message) error {
	start := time.Now()
	err := m.Publisher.Publish(ctx, msg)
	log.Printf("[Publish] topic=%s key=%s took=%s err=%v", msg.Topic, msg.Key, time.Since(start), err)
	return err
}

func (m *LoggingMiddleware) Subscribe(ctx context.Context, topic string, handler func(Message) error) error {
	wrapped := func(msg Message) error {
		start := time.Now()
		err := handler(msg)
		log.Printf("[Consume] topic=%s key=%s took=%s err=%v", msg.Topic, msg.Key, time.Since(start), err)
		return err
	}
	return m.Subscriber.Subscribe(ctx, topic, wrapped)
}
