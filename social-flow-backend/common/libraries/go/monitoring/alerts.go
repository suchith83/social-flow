package monitoring

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// Simple PagerDuty-compatible event payload (v2 events API).
type PagerDutyEvent struct {
	RoutingKey  string `json:"routing_key"`
	EventAction string `json:"event_action"` // "trigger" | "acknowledge" | "resolve"
	DedupKey    string `json:"dedup_key,omitempty"`
	Payload     struct {
		Summary   string `json:"summary"`
		Source    string `json:"source"`
		Severity  string `json:"severity"` // "critical","error","warning","info"
		Component string `json:"component,omitempty"`
	} `json:"payload"`
}

// PagerDutyClient sends events to PD.
type PagerDutyClient struct {
	APIKey string
	Url    string // default https://events.pagerduty.com/v2/enqueue
	Client *http.Client
	Logger *zap.Logger
}

// NewPagerDutyClient constructs a client.
func NewPagerDutyClient(apiKey string, logger *zap.Logger) *PagerDutyClient {
	url := "https://events.pagerduty.com/v2/enqueue"
	return &PagerDutyClient{
		APIKey: apiKey,
		Url:    url,
		Client: &http.Client{Timeout: 5 * time.Second},
		Logger: logger,
	}
}

// TriggerAlert sends a trigger event.
func (p *PagerDutyClient) TriggerAlert(ctx context.Context, summary, source, severity, component, dedupKey string) error {
	ev := PagerDutyEvent{
		RoutingKey:  p.APIKey,
		EventAction: "trigger",
		DedupKey:    dedupKey,
	}
	ev.Payload.Summary = summary
	ev.Payload.Source = source
	ev.Payload.Severity = severity
	ev.Payload.Component = component

	body, err := json.Marshal(ev)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.Url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		if p.Logger != nil {
			p.Logger.Error("pagerduty.request_failed", zap.Error(err))
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		err = fmt.Errorf("pagerduty responded %d", resp.StatusCode)
		if p.Logger != nil {
			p.Logger.Warn("pagerduty.bad_status", zap.Int("status", resp.StatusCode))
		}
		return err
	}

	if p.Logger != nil {
		p.Logger.Info("pagerduty.alert_sent", zap.String("summary", summary))
	}
	return nil
}
