package monitoring

import (
	"context"
	"net/http"
	"strings"
	"time"

	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// HTTPMiddlewareConfig bundles dependencies for middleware.
type HTTPMiddlewareConfig struct {
	Logger   *zap.Logger
	Counter  *prometheus.CounterVec
	Histogram *prometheus.HistogramVec
	Tracer   trace.TracerProvider
	ServiceName string
}

// InstrumentationMiddleware returns a net/http middleware that records Prometheus metrics,
// starts an OpenTelemetry span (if tracer present), and logs request/response summary.
func InstrumentationMiddleware(cfg HTTPMiddlewareConfig, handlerName string) func(http.Handler) http.Handler {
	tracer := cfg.Tracer.Tracer(handlerName)
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			lrw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

			// create context with span if tracer available
			ctx := r.Context()
			var span trace.Span
			if tracer != nil {
				ctx, span = tracer.Start(ctx, strings.Join([]string{handlerName, r.Method, r.URL.Path}, " "))
				defer span.End()
			}

			// log incoming request
			if cfg.Logger != nil {
				cfg.Logger.Info("http.request.start",
					zap.String("method", r.Method),
					zap.String("path", r.URL.Path),
					zap.String("remote", r.RemoteAddr),
				)
			}

			next.ServeHTTP(lrw, r.WithContext(ctx))

			duration := time.Since(start)
			status := lrw.statusCodeString()

			// observe metrics
			if cfg.Counter != nil && cfg.Histogram != nil {
				ObserveRequest(cfg.Counter, cfg.Histogram, handlerName, r.Method, status, duration)
			}

			// log response summary
			if cfg.Logger != nil {
				cfg.Logger.Info("http.request.done",
					zap.String("method", r.Method),
					zap.String("path", r.URL.Path),
					zap.Int("status", lrw.statusCode),
					zap.Duration("duration", duration),
				)
			}
		})
	}
}

// responseWriter is helper to capture status codes.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) statusCodeString() string {
	return http.StatusText(rw.statusCode)
}

// Gin helper: returns gin.HandlerFunc that uses the same InstrumentationMiddleware concepts.
// This is optional convenience for projects using gin.
