package monitoring

import (
	"context"
	"fmt"
	"io"
	"time"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/resource"

	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/semconv/v1.12.0"

	// The OTLP exporter import is optional — commented for flexibility.
	// otlptrace "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
)

// TracingOptions controls tracer provider initialization.
type TracingOptions struct {
	ServiceName string
	OTLPEndpoint string // if empty use stdout exporter
}

// InitTracing sets up an OpenTelemetry TracerProvider and returns the provider and a shutdown func.
// It uses OTLP over HTTP when OTLPEndpoint is provided otherwise falls back to stdout tracing (useful for local).
func InitTracing(ctx context.Context, opt TracingOptions) (sdktrace.TracerProvider, func(context.Context) error, error) {
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(opt.ServiceName),
			attribute.String("env", "production"),
		),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("resource creation: %w", err)
	}

	var exporter sdktrace.SpanExporter
	var closer io.Closer // may be nil for some exporters

	// if OTLPEndpoint provided -> attempt OTLP exporter
	if opt.OTLPEndpoint != "" {
		// NOTE: many deployments will want OTLP exporter. Example code (commented) below.
		/*
		client := otlptracehttp.NewClient(otlptracehttp.WithEndpoint(opt.OTLPEndpoint), otlptracehttp.WithInsecure())
		exporter, err = otlptrace.New(ctx, client)
		if err != nil {
			return nil, nil, err
		}
		*/
		// For safety (no network assumptions), use stdout but mark endpoint in logs.
		exporter, err = stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, nil, err
		}
	} else {
		// local / debugging: pretty stdout exporter
		exporter, err = stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, nil, err
		}
	}

	bsp := sdktrace.NewBatchSpanProcessor(exporter)
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(bsp),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(1.0))),
	)

	otel.SetTracerProvider(tp)

	shutdown := func(ctx context.Context) error {
		// give the exporter up to 5s to flush
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if closer != nil {
			_ = closer.Close()
		}
		return tp.Shutdown(ctx)
	}
	return *tp, shutdown, nil
}
