// ========================================
// File: nats.go
// ========================================
package messaging

import (
	"context"

	"github.com/nats-io/nats.go"
)

type NATS struct {
	conn *nats.Conn
}

func NewNATS(url string) (*NATS, error) {
	nc, err := nats.Connect(url)
	if err != nil {
		return nil, err
	}
	return &NATS{conn: nc}, nil
}

func (n *NATS) Publish(ctx context.Context, msg Message) error {
	return n.conn.Publish(msg.Topic, msg.Value)
}

func (n *NATS) Subscribe(ctx context.Context, topic string, handler func(Message) error) error {
	_, err := n.conn.Subscribe(topic, func(m *nats.Msg) {
		_ = handler(Message{
			Topic:   topic,
			Value:   m.Data,
			Headers: map[string]string{},
		})
	})
	return err
}

func (n *NATS) Close() error {
	n.conn.Drain()
	n.conn.Close()
	return nil
}
