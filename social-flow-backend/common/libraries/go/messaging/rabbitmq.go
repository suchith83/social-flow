// ========================================
// File: rabbitmq.go
// ========================================
package messaging

import (
	"context"
	"encoding/json"
	"log"

	amqp "github.com/rabbitmq/amqp091-go"
)

type RabbitMQ struct {
	conn    *amqp.Connection
	channel *amqp.Channel
}

func NewRabbitMQ(dsn string) (*RabbitMQ, error) {
	conn, err := amqp.Dial(dsn)
	if err != nil {
		return nil, err
	}
	ch, err := conn.Channel()
	if err != nil {
		return nil, err
	}
	return &RabbitMQ{conn: conn, channel: ch}, nil
}

func (r *RabbitMQ) Publish(ctx context.Context, msg Message) error {
	body, err := json.Marshal(msg.Value)
	if err != nil {
		return err
	}
	return r.channel.PublishWithContext(ctx, "", msg.Topic, false, false, amqp.Publishing{
		ContentType: "application/json",
		Body:        body,
	})
}

func (r *RabbitMQ) Subscribe(ctx context.Context, queue string, handler func(Message) error) error {
	msgs, err := r.channel.Consume(queue, "", true, false, false, false, nil)
	if err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case d := <-msgs:
			m := Message{
				Topic: queue,
				Value: d.Body,
			}
			if err := handler(m); err != nil {
				log.Printf("⚠️ handler error: %v", err)
			}
		}
	}
}

func (r *RabbitMQ) Close() error {
	if err := r.channel.Close(); err != nil {
		return err
	}
	return r.conn.Close()
}
