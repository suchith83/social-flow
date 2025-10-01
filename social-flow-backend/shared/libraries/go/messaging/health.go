// ========================================
// File: health.go
// ========================================
package messaging

import "context"

// HealthCheck verifies messaging system
func HealthCheck(ctx context.Context, pub Publisher, sub Subscriber) error {
	testMsg := Message{
		Topic:     "healthcheck",
		Key:       "ping",
		Value:     []byte("pong"),
		Timestamp: time.Now(),
	}
	if err := pub.Publish(ctx, testMsg); err != nil {
		return err
	}
	return nil
}
