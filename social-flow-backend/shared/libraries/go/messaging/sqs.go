// ========================================
// File: sqs.go
// ========================================
package messaging

import (
	"context"
	"encoding/json"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/sqs"
)

type SQS struct {
	client *sqs.Client
	url    string
}

func NewSQS(client *sqs.Client, queueURL string) *SQS {
	return &SQS{client: client, url: queueURL}
}

func (s *SQS) Publish(ctx context.Context, msg Message) error {
	body, _ := json.Marshal(msg)
	_, err := s.client.SendMessage(ctx, &sqs.SendMessageInput{
		QueueUrl:    &s.url,
		MessageBody: aws.String(string(body)),
	})
	return err
}

func (s *SQS) Subscribe(ctx context.Context, topic string, handler func(Message) error) error {
	for {
		out, err := s.client.ReceiveMessage(ctx, &sqs.ReceiveMessageInput{
			QueueUrl:            &s.url,
			MaxNumberOfMessages: 5,
			WaitTimeSeconds:     10,
		})
		if err != nil {
			return err
		}
		for _, m := range out.Messages {
			msg := Message{
				Topic:     topic,
				Value:     []byte(*m.Body),
				Timestamp: time.Now(),
			}
			_ = handler(msg)
		}
	}
}
func (s *SQS) Close() error { return nil }
