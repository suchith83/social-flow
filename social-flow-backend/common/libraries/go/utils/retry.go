package utils

import (
	"context"
	"math"
	"math/rand"
	"time"
)

// RetryConfig controls the retry behavior.
type RetryConfig struct {
	MaxAttempts int
	BaseDelay   time.Duration // initial delay
	MaxDelay    time.Duration // maximum backoff
	Multiplier  float64       // exponential multiplier
	Jitter      bool          // add jitter
}

// DefaultRetryConfig provides a sane default exponential backoff.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts: 5,
		BaseDelay:   100 * time.Millisecond,
		MaxDelay:    5 * time.Second,
		Multiplier:  2.0,
		Jitter:      true,
	}
}

// RetryWithContext retries the provided operation until it succeeds or context is done or attempts exhausted.
// The operation should return nil on success, or an error that determines whether to retry.
// If operation returns ErrDoNotRetry (special sentinel), retry stops immediately.
var ErrDoNotRetry = &doNotRetryError{}

type doNotRetryError struct{}

func (d *doNotRetryError) Error() string { return "do not retry" }
func DoNotRetry() error                  { return ErrDoNotRetry }

// RetryWithContext executes op up to cfg.MaxAttempts times.
func RetryWithContext(ctx context.Context, cfg RetryConfig, op func() error) error {
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 1
	}
	if cfg.BaseDelay <= 0 {
		cfg.BaseDelay = 100 * time.Millisecond
	}
	if cfg.MaxDelay <= 0 {
		cfg.MaxDelay = 5 * time.Second
	}
	if cfg.Multiplier <= 0 {
		cfg.Multiplier = 2.0
	}

	var lastErr error
	delay := float64(cfg.BaseDelay)

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		lastErr = op()
		if lastErr == nil {
			return nil
		}
		// immediate stop
		if lastErr == ErrDoNotRetry {
			return lastErr
		}

		// if this was the last attempt, break
		if attempt == cfg.MaxAttempts {
			break
		}

		// calculate next delay
		sleep := time.Duration(delay)
		if cfg.Jitter {
			// jitter: uniform in [delay/2, delay*1.5]
			min := delay / 2
			max := delay * 1.5
			sleep = time.Duration(min + rand.Float64()*(max-min))
		}
		if sleep > cfg.MaxDelay {
			sleep = cfg.MaxDelay
		}

		timer := time.NewTimer(sleep)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}

		// increase delay
		delay = math.Min(delay*cfg.Multiplier, float64(cfg.MaxDelay))
	}
	return lastErr
}
