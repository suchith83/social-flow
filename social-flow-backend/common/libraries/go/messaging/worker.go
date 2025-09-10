// ========================================
// File: worker.go
// ========================================
package messaging

import (
	"context"
	"log"
	"time"
)

// Worker consumes messages in background
type Worker struct {
	subscriber Subscriber
	topic      string
	handler    func(Message) error
}

func NewWorker(sub Subscriber, topic string, handler func(Message) error) *Worker {
	return &Worker{subscriber: sub, topic: topic, handler: handler}
}

func (w *Worker) Start(ctx context.Context) {
	go func() {
		for {
			if err := w.subscriber.Subscribe(ctx, w.topic, w.handler); err != nil {
				log.Printf("⚠️ worker error: %v", err)
			}
			time.Sleep(2 * time.Second)
		}
	}()
}
