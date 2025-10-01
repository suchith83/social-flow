package utils

import (
	"regexp"
	"strings"
)

// Precompiled regexps
var (
	emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$`)
	uuidRegex  = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$`)
)

// IsEmail checks whether the string is a valid-looking email address.
func IsEmail(s string) bool {
	s = strings.TrimSpace(s)
	return emailRegex.MatchString(s)
}

// IsUUID validates a canonical RFC 4122 UUID string.
func IsUUID(s string) bool {
	return uuidRegex.MatchString(s)
}

// TrimAndLower is a small helper used commonly when normalizing keys/usernames.
func TrimAndLower(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}
