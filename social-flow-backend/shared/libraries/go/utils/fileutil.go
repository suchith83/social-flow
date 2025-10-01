package utils

import (
	"io"
	"os"
	"path/filepath"
)

// EnsureDir ensures a directory exists (creates with 0o750 if missing).
func EnsureDir(path string) error {
	if path == "" {
		return nil
	}
	return os.MkdirAll(path, 0o750)
}

// FileExists returns true if file exists and is not a directory.
func FileExists(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !fi.IsDir()
}

// ReadFileSafe reads a file fully and returns bytes. Uses os.ReadFile under hood.
func ReadFileSafe(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// WriteFileAtomic writes to a temporary file and renames it into place for atomic updates.
// Creates parent directories as needed.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := EnsureDir(dir); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() {
		tmp.Close()
		_ = os.Remove(tmpName)
	}()

	if _, err := io.Copy(tmp, bytesReader(data)); err != nil { // use helper to avoid importing bytes repeatedly
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// helper to avoid adding bytes import above
func bytesReader(b []byte) io.Reader { return &byteReader{b: b} }

type byteReader struct{ b []byte }

func (r *byteReader) Read(p []byte) (int, error) {
	if len(r.b) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.b)
	r.b = r.b[n:]
	return n, nil
}
