package utils

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"sync"
)

// LoadJSONConfig loads JSON config from a file path into the provided pointer `out`.
// `out` must be a pointer to a struct or map. Returns an error if reading/parsing fails.
func LoadJSONConfig(path string, out interface{}) error {
	if out == nil {
		return errors.New("nil output provided to LoadJSONConfig")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return DecodeJSONFromReader(f, out)
}

// DecodeJSONFromReader decodes JSON from any io.Reader into out. Useful for testing.
func DecodeJSONFromReader(r io.Reader, out interface{}) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields() // helps catch mis-typed keys
	return dec.Decode(out)
}

var (
	envOnce sync.Once
	envMap  map[string]string
)

// LoadEnv loads all environment variables into an in-memory map (cached).
// Useful for deterministic reads in tests if you set envMap directly.
func LoadEnv() map[string]string {
	envOnce.Do(func() {
		envMap = map[string]string{}
		for _, kv := range os.Environ() {
			// split at first '='
			for i := 0; i < len(kv); i++ {
				if kv[i] == '=' {
					envMap[kv[:i]] = kv[i+1:]
					break
				}
			}
		}
	})
	return envMap
}
