// ========================================
// File: kafka.go
// ========================================
package messaging

import (
	"context"
	"fmt"

	kafka "github.com/segmentio/kafka-go"
)

// KafkaPublisher implements Publisher
type KafkaPublisher struct {
	writer *kafka.Writer
}

func NewKafkaPublisher(brokers []string, topic string) *KafkaPublisher {
	return &KafkaPublisher{
		writer: &kafka.Writer{
			Addr:     kafka.TCP(brokers...),
			Topic:    topic,
			Balancer: &kafka.LeastBytes{},
		},
	}
}

func (k *KafkaPublisher) Publish(ctx context.Context, msg Message) error {
	headers := []kafka.Header{}
	for k, v := range msg.Headers {
		headers = append(headers, kafka.Header{Key: k, Value: []byte(v)})
	}
	return k.writer.WriteMessages(ctx, kafka.Message{
		Key:     []byte(msg.Key),
		Value:   msg.Value,
		Time:    msg.Timestamp,
		Headers: headers,
	})
}

func (k *KafkaPublisher) Close() error {
	return k.writer.Close()
}

// KafkaSubscriber implements Subscriber
type KafkaSubscriber struct {
	reader *kafka.Reader
}

func NewKafkaSubscriber(brokers []string, groupID, topic string) *KafkaSubscriber {
	return &KafkaSubscriber{
		reader: kafka.NewReader(kafka.ReaderConfig{
			Brokers: brokers,
			GroupID: groupID,
			Topic:   topic,
		}),
	}
}

func (k *KafkaSubscriber) Subscribe(ctx context.Context, topic string, handler func(Message) error) error {
	for {
		m, err := k.reader.ReadMessage(ctx)
		if err != nil {
			return err
		}
		msg := Message{
			Topic:     m.Topic,
			Key:       string(m.Key),
			Value:     m.Value,
			Timestamp: m.Time,
			Headers:   map[string]string{},
		}
		for _, h := range m.Headers {
			msg.Headers[h.Key] = string(h.Value)
		}
		if err := handler(msg); err != nil {
			fmt.Printf("⚠️ Error handling message: %v\n", err)
		}
	}
}

func (k *KafkaSubscriber) Close() error {
	return k.reader.Close()
}
