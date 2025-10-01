package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"time"
)

// HTTPClientConfig allows customizing the client behavior.
type HTTPClientConfig struct {
	Timeout       time.Duration
	IdleConnTimeout time.Duration
	MaxIdleConns  int
	KeepAlive     time.Duration
}

// NewHTTPClient builds a performant http.Client with sensible defaults.
func NewHTTPClient(cfg *HTTPClientConfig) *http.Client {
	if cfg == nil {
		cfg = &HTTPClientConfig{
			Timeout:         10 * time.Second,
			IdleConnTimeout: 90 * time.Second,
			MaxIdleConns:    100,
			KeepAlive:       30 * time.Second,
		}
	}
	transport := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   cfg.Timeout,
			KeepAlive: cfg.KeepAlive,
		}).DialContext,
		IdleConnTimeout:     cfg.IdleConnTimeout,
		MaxIdleConns:        cfg.MaxIdleConns,
		TLSHandshakeTimeout: 10 * time.Second,
	}
	return &http.Client{
		Transport: transport,
		Timeout:   cfg.Timeout,
	}
}

// DoJSON performs an HTTP request with a JSON body and decodes a JSON response into out.
// If out is nil the response body is discarded. Accepts context for cancellation.
// Returns response status code and error.
func DoJSON(ctx context.Context, client *http.Client, method, url string, headers map[string]string, body interface{}, out interface{}) (int, error) {
	var bodyReader io.Reader
	if body != nil {
		bs, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		bodyReader = bytes.NewReader(bs)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	// On non-2xx treat as error but still provide body (helpful for debugging)
	if out == nil {
		// discard
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.StatusCode, nil
	}

	dec := json.NewDecoder(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if err := dec.Decode(out); err != nil && !errors.Is(err, io.EOF) {
			return resp.StatusCode, err
		}
		return resp.StatusCode, nil
	}

	// try to decode error payload into a generic map for caller
	var payload map[string]interface{}
	_ = dec.Decode(&payload)
	return resp.StatusCode, errors.New("non-2xx response received")
}
