package redis

import (
	"context"
	"fmt"
	"sync"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// PubSubHandler is a callback invoked for each message
type PubSubHandler func(channel string, payload string)

// Subscriber holds subscription state
type Subscriber struct {
	client redis.UniversalClient
	mu     sync.Mutex
	// subscriptions map channel->handler
	subs map[string]PubSubHandler
	ps   *redis.PubSub
}

// NewSubscriber creates a new Subscriber using the global Client
func NewSubscriber() (*Subscriber, error) {
	if Client == nil {
		return nil, fmt.Errorf("redis client not initialized")
	}
	// Client may not expose Subscribe method via Cmdable; cast to UniversalClient
	c, ok := Client.(redis.UniversalClient)
	if !ok {
		return nil, fmt.Errorf("client does not support pubsub")
	}
	return &Subscriber{
		client: c,
		subs:   make(map[string]PubSubHandler),
	}, nil
}

// Subscribe subscribes to channels and runs handlers in background goroutines.
// It will attempt reconnection on errors.
func (s *Subscriber) Subscribe(ctx context.Context, channel string, handler PubSubHandler) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.subs[channel]; exists {
		// already subscribed; replace handler
		s.subs[channel] = handler
		return nil
	}
	if s.ps == nil {
		s.ps = s.client.Subscribe(ctx) // start with empty subscription
	}
	if err := s.ps.Subscribe(ctx, channel); err != nil {
		return err
	}
	s.subs[channel] = handler

	// Start reader goroutine once
	go s.readerLoop()
	return nil
}

// readerLoop listens for messages and dispatches to handlers.
func (s *Subscriber) readerLoop() {
	for {
		msg, err := s.ps.ReceiveMessage(context.Background())
		if err != nil {
			// handle reconnection/backoff
			time.Sleep(500 * time.Millisecond)
			continue
		}
		s.mu.Lock()
		handler := s.subs[msg.Channel]
		s.mu.Unlock()
		if handler != nil {
			go handler(msg.Channel, msg.Payload)
		}
	}
}

// Publish helper
func Publish(ctx context.Context, channel string, payload string) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	return Client.Publish(ctx, channel, payload).Err()
}
