package monitoring

import (
	"context"
	"net/http"
	"sync/atomic"
	"time"
)

// HealthServer maintains basic liveness/readiness flags and endpoints.
type HealthServer struct {
	live      atomic.Bool
	ready     atomic.Bool
	mux       *http.ServeMux
	server    *http.Server
}

// NewHealthServer prepares a server that serves /live and /ready endpoints.
func NewHealthServer(bindAddr string) *HealthServer {
	mux := http.NewServeMux()
	hs := &HealthServer{
		mux: mux,
		server: &http.Server{
			Addr:    bindAddr,
			Handler: mux,
		},
	}
	mux.HandleFunc("/live", hs.livenessHandler)
	mux.HandleFunc("/ready", hs.readinessHandler)
	hs.SetLiveness(true)
	hs.SetReadiness(false)
	return hs
}

func (h *HealthServer) SetLiveness(v bool) {
	h.live.Store(v)
}

func (h *HealthServer) SetReadiness(v bool) {
	h.ready.Store(v)
}

func (h *HealthServer) livenessHandler(w http.ResponseWriter, r *http.Request) {
	if h.live.Load() {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
		return
	}
	http.Error(w, "not live", http.StatusServiceUnavailable)
}

func (h *HealthServer) readinessHandler(w http.ResponseWriter, r *http.Request) {
	if h.ready.Load() {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ready"))
		return
	}
	http.Error(w, "not ready", http.StatusServiceUnavailable)
}

// Start runs the health server in the background; call Shutdown to stop it.
func (h *HealthServer) Start(ctx context.Context) error {
	go func() {
		_ = h.server.ListenAndServe()
	}()
	// Wait briefly to allow listen, returning nil immediately (non-blocking)
	return nil
}

// Shutdown gracefully stops the health server.
func (h *HealthServer) Shutdown(ctx context.Context) error {
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return h.server.Shutdown(shutdownCtx)
}
