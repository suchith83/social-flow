package utils

import (
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"
)

type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Logger provides a minimal structured logger wrapper around the standard library log.Logger.
// It's intentionally simple to avoid external deps; replace with zap/logrus if desired.
type Logger struct {
	mu     sync.Mutex
	out    io.Writer
	prefix string
	level  Level
	std    *log.Logger
}

// global singleton
var (
	defaultLogger *Logger
	loggerOnce    sync.Once
)

// NewLogger constructs a Logger writing to out with prefix and minimum level.
func NewLogger(out io.Writer, prefix string, level Level) *Logger {
	l := &Logger{
		out:    out,
		prefix: prefix,
		level:  level,
		std:    log.New(out, "", 0), // we format timestamp ourselves
	}
	return l
}

// GetLogger returns a process-wide logger singleton (initialized on first call).
// By default it logs to stderr, LevelInfo.
func GetLogger() *Logger {
	loggerOnce.Do(func() {
		defaultLogger = NewLogger(os.Stderr, "", LevelInfo)
	})
	return defaultLogger
}

func (l *Logger) logf(level Level, format string, args ...interface{}) {
	if level < l.level {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	ts := time.Now().UTC().Format(time.RFC3339Nano)
	msg := fmt.Sprintf(format, args...)
	l.std.Printf("%s %s %s %s", ts, level.String(), l.prefix, msg)
}

// Convenience wrappers:
func (l *Logger) Debugf(format string, args ...interface{}) { l.logf(LevelDebug, format, args...) }
func (l *Logger) Infof(format string, args ...interface{})  { l.logf(LevelInfo, format, args...) }
func (l *Logger) Warnf(format string, args ...interface{})  { l.logf(LevelWarn, format, args...) }
func (l *Logger) Errorf(format string, args ...interface{}) { l.logf(LevelError, format, args...) }

// SetLevel changes the logger's minimum level.
func (l *Logger) SetLevel(level Level) { l.mu.Lock(); l.level = level; l.mu.Unlock() }
