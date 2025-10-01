package utils

import (
	"errors"
	"fmt"
)

// Common sentinel errors
var (
	ErrNotFound     = errors.New("not found")
	ErrInvalidInput = errors.New("invalid input")
	ErrTimeout      = errors.New("timeout")
)

// Wrapf wraps an error with a formatted message while preserving the original error.
func Wrapf(err error, format string, args ...interface{}) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), err)
}

// Is reports whether target error is in err's chain.
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As tries to cast err into target type.
func As(err error, target interface{}) bool {
	return errors.As(err, target)
}
