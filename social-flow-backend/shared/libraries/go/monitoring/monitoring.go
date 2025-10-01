package monitoring

import (
	"context"
	"time"

	"go.uber.org/zap"
	"go.opentelemetry.io/otel/trace"
	"github.com/prometheus/client_golang/prometheus"
)

// MonitoringManager centralizes monitoring concerns.
type MonitoringManager struct {
	Logger   *zap.Logger
	Registry *prometheus.Registry
	Tracer   trace.TracerProvider
	// convenience metric handles
	RequestCounter   prometheus.CounterVec
	RequestHistogram prometheus.HistogramVec
	// config
	cfg Config
}

// Config controls initialization behaviour.
type Config struct {
	ServiceName        string
	PrometheusEndpoint string // host:port to expose metrics; if empty user can mount handler manually
	EnablePrometheus   bool
	EnableTracing      bool
	OTLPEndpoint       string // if empty, tracing will use stdout exporter for debugging
	EnableLogging      bool
	LogLevel           string // "debug","info","warn","error"
}
