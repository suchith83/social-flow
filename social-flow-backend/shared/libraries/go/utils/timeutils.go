package utils

import (
	"context"
	"time"
)

// NowUTC returns current time in UTC.
func NowUTC() time.Time { return time.Now().UTC() }

// ParseRFC3339OrUnix tries to parse s as RFC3339, then as unix seconds integer.
func ParseRFC3339OrUnix(s string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339, s)
	if err == nil {
		return t, nil
	}
	// try unix seconds
	var secs int64
	_, err2 := fmt.Sscanf(s, "%d", &secs)
	if err2 == nil {
		return time.Unix(secs, 0).UTC(), nil
	}
	return time.Time{}, err
}

// SleepContext sleeps for d or until context is canceled (better than time.Sleep for servers).
func SleepContext(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
