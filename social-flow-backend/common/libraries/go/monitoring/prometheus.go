package monitoring

import (
	"context"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// NewPrometheusRegistry creates a fresh registry with standard process/go collectors.
func NewPrometheusRegistry() *prometheus.Registry {
	reg := prometheus.NewRegistry()
	// add process and go metrics which are useful in production
	reg.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	reg.MustRegister(prometheus.NewGoCollector())
	return reg
}

// DefaultMetricVecs registers the commonly-used request metrics.
func DefaultMetricVecs(reg *prometheus.Registry, namespace string) (prometheus.CounterVec, prometheus.HistogramVec) {
	reqCounter := *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "http_requests_total",
			Help:      "Total number of HTTP requests processed, partitioned by status and method.",
		},
		[]string{"handler", "method", "status"},
	)

	reqHistogram := *prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "http_request_duration_seconds",
			Help:      "HTTP request duration in seconds.",
			// buckets suited for HTTP: from ms to seconds
			Buckets: prometheus.DefBuckets,
		},
		[]string{"handler", "method"},
	)

	reg.MustRegister(&reqCounter)
	reg.MustRegister(&reqHistogram)

	return reqCounter, reqHistogram
}

// MetricsHandler returns an http.Handler to serve metrics (usable with http.Server or mux)
func MetricsHandler(reg *prometheus.Registry) http.Handler {
	return promhttp.HandlerFor(reg, promhttp.HandlerOpts{})
}

// ExposeMetricsServe starts a small HTTP server that only serves /metrics and readiness routes.
// Use for when you want a dedicated metrics port (common in k8s sidecar patterns).
func ExposeMetricsServe(ctx context.Context, addr string, reg *prometheus.Registry, readiness func() bool) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", MetricsHandler(reg))
	if readiness != nil {
		mux.HandleFunc("/ready", func(w http.ResponseWriter, r *http.Request) {
			if readiness() {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("ok"))
				return
			}
			http.Error(w, "not ready", http.StatusServiceUnavailable)
		})
	}
	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	// shutdown goroutine
	go func() {
		<-ctx.Done()
		_ = server.Close()
	}()
	return server.ListenAndServe()
}

// ObserveRequest is helper for recording metrics for a single request.
func ObserveRequest(counter *prometheus.CounterVec, hist *prometheus.HistogramVec, handler, method, status string, duration time.Duration) {
	if counter != nil {
		counter.WithLabelValues(handler, method, status).Inc()
	}
	if hist != nil {
		hist.WithLabelValues(handler, method).Observe(duration.Seconds())
	}
}
