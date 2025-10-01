package security

import (
	"regexp"
	"strings"
)

// SanitizeInput removes dangerous characters to prevent injection attacks.
func SanitizeInput(input string) string {
	re := regexp.MustCompile(`['"\\;]`)
	return re.ReplaceAllString(input, "")
}

// NormalizeEmail lowercases and trims spaces.
func NormalizeEmail(email string) string {
	return strings.TrimSpace(strings.ToLower(email))
}
