# Monitoring and Observability Guide

This guide covers comprehensive monitoring, logging, metrics, and observability for the Social Flow Backend to ensure system health, performance, and reliability.

## 📊 Monitoring Overview

### Monitoring Stack

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger + OpenTelemetry
- **APM**: AWS X-Ray
- **Alerting**: AlertManager + PagerDuty
- **Uptime**: Pingdom + UptimeRobot

### Key Metrics Categories

1. **Application Metrics**: Request rates, response times, error rates
2. **Infrastructure Metrics**: CPU, memory, disk, network
3. **Business Metrics**: User engagement, content performance, revenue
4. **Security Metrics**: Failed logins, suspicious activity, rate limiting
5. **Database Metrics**: Query performance, connection pools, replication lag

## 🔧 Prometheus Configuration

### Prometheus Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'socialflow-backend'
    static_configs:
      - targets: ['socialflow-backend:8000']
    metrics_path: /metrics
    scrape_interval: 5s
    scrape_timeout: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s
```

### Application Metrics

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from fastapi import Request, Response
import time

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

REQUEST_SIZE = Summary(
    'http_request_size_bytes',
    'HTTP request size',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Summary(
    'http_response_size_bytes',
    'HTTP response size',
    ['method', 'endpoint']
)

# Business metrics
USER_REGISTRATIONS = Counter(
    'user_registrations_total',
    'Total user registrations'
)

VIDEO_UPLOADS = Counter(
    'video_uploads_total',
    'Total video uploads',
    ['status']
)

VIDEO_VIEWS = Counter(
    'video_views_total',
    'Total video views',
    ['video_id', 'user_id']
)

POST_LIKES = Counter(
    'post_likes_total',
    'Total post likes',
    ['post_id', 'user_id']
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

REDIS_CONNECTIONS = Gauge(
    'redis_connections_active',
    'Number of active Redis connections'
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

# Custom metrics
def record_request_metrics(request: Request, response: Response, duration: float):
    """Record request metrics."""
    method = request.method
    endpoint = request.url.path
    status_code = str(response.status_code)
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    if hasattr(request, 'content_length') and request.content_length:
        REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(request.content_length)
    
    if hasattr(response, 'content_length') and response.content_length:
        RESPONSE_SIZE.labels(method=method, endpoint=endpoint).observe(response.content_length)

def record_business_metrics(event_type: str, **kwargs):
    """Record business metrics."""
    if event_type == 'user_registration':
        USER_REGISTRATIONS.inc()
    elif event_type == 'video_upload':
        status = kwargs.get('status', 'unknown')
        VIDEO_UPLOADS.labels(status=status).inc()
    elif event_type == 'video_view':
        video_id = kwargs.get('video_id', 'unknown')
        user_id = kwargs.get('user_id', 'unknown')
        VIDEO_VIEWS.labels(video_id=video_id, user_id=user_id).inc()
    elif event_type == 'post_like':
        post_id = kwargs.get('post_id', 'unknown')
        user_id = kwargs.get('user_id', 'unknown')
        POST_LIKES.labels(post_id=post_id, user_id=user_id).inc()

def record_error_metrics(error_type: str, endpoint: str):
    """Record error metrics."""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

def get_metrics():
    """Get Prometheus metrics."""
    return generate_latest()
```

### Metrics Middleware

```python
# app/middleware/metrics.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from app.core.metrics import record_request_metrics, record_error_metrics

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            record_request_metrics(request, response, duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            endpoint = request.url.path
            
            record_error_metrics(error_type, endpoint)
            
            raise
```

## 📈 Grafana Dashboards

### Application Dashboard

```json
{
  "dashboard": {
    "title": "Social Flow Backend - Application",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          },
          {
            "expr": "rate(http_requests_total{status_code=~\"4..\"}[5m])",
            "legendFormat": "4xx errors"
          }
        ],
        "yAxes": [
          {
            "label": "Errors/sec"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(rate(user_registrations_total[1h]))",
            "legendFormat": "New users/hour"
          }
        ]
      },
      {
        "title": "Video Uploads",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(video_uploads_total[5m])",
            "legendFormat": "{{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Uploads/sec"
          }
        ]
      },
      {
        "title": "Video Views",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(video_views_total[5m])",
            "legendFormat": "Views/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Views/sec"
          }
        ]
      }
    ]
  }
}
```

### Infrastructure Dashboard

```json
{
  "dashboard": {
    "title": "Social Flow Backend - Infrastructure",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{instance}}"
          }
        ],
        "yAxes": [
          {
            "label": "CPU %",
            "max": 100
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "{{instance}}"
          }
        ],
        "yAxes": [
          {
            "label": "Memory %",
            "max": 100
          }
        ]
      },
      {
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)",
            "legendFormat": "{{instance}} {{mountpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Disk %",
            "max": 100
          }
        ]
      },
      {
        "title": "Network I/O",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total[5m])",
            "legendFormat": "{{instance}} {{device}} RX"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total[5m])",
            "legendFormat": "{{instance}} {{device}} TX"
          }
        ],
        "yAxes": [
          {
            "label": "Bytes/sec"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "database_connections_active",
            "legendFormat": "Active connections"
          }
        ],
        "yAxes": [
          {
            "label": "Connections"
          }
        ]
      },
      {
        "title": "Redis Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_connections_active",
            "legendFormat": "Active connections"
          }
        ],
        "yAxes": [
          {
            "label": "Connections"
          }
        ]
      }
    ]
  }
}
```

## 📝 Logging Configuration

### Structured Logging

```python
# app/core/logging.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

class CustomJSONFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = 'socialflow-backend'
        log_record['version'] = '1.0.0'
        log_record['environment'] = 'production'
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id

def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging."""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create JSON formatter
    json_formatter = CustomJSONFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    console_handler.setFormatter(json_formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

class LoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        return msg, {**kwargs, **self.extra}

def get_logger(name: str, **extra: Any) -> LoggerAdapter:
    """Get logger with extra context."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, extra)
```

### Logging Middleware

```python
# app/middleware/logging.py
import uuid
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import get_logger

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Add to request state
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        
        # Create logger with context
        logger = get_logger(
            __name__,
            request_id=request_id,
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )
        
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "event": "request_started",
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers)
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "event": "request_completed",
                    "status_code": response.status_code,
                    "duration": duration,
                    "response_headers": dict(response.headers)
                }
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "event": "request_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration
                },
                exc_info=True
            )
            
            raise
```

### Business Event Logging

```python
# app/core/events.py
from typing import Any, Dict, Optional
from app.core.logging import get_logger

class EventLogger:
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def log_user_event(self, event_type: str, user_id: str, **data: Any) -> None:
        """Log user-related events."""
        self.logger.info(
            f"User event: {event_type}",
            extra={
                "event_type": event_type,
                "user_id": user_id,
                "event_category": "user",
                **data
            }
        )
    
    def log_content_event(self, event_type: str, content_id: str, user_id: Optional[str] = None, **data: Any) -> None:
        """Log content-related events."""
        self.logger.info(
            f"Content event: {event_type}",
            extra={
                "event_type": event_type,
                "content_id": content_id,
                "user_id": user_id,
                "event_category": "content",
                **data
            }
        )
    
    def log_system_event(self, event_type: str, **data: Any) -> None:
        """Log system events."""
        self.logger.info(
            f"System event: {event_type}",
            extra={
                "event_type": event_type,
                "event_category": "system",
                **data
            }
        )
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None, **data: Any) -> None:
        """Log security events."""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": event_type,
                "user_id": user_id,
                "event_category": "security",
                **data
            }
        )

# Global event logger instance
event_logger = EventLogger()
```

## 🔍 Distributed Tracing

### OpenTelemetry Configuration

```python
# app/core/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import os

def setup_tracing(app) -> None:
    """Setup distributed tracing."""
    
    # Create resource
    resource = Resource.create({
        "service.name": "socialflow-backend",
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })
    
    # Create tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    
    # Create Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", "14268")),
    )
    
    # Create span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()
    
    # Instrument Redis
    RedisInstrumentor().instrument()
    
    # Instrument HTTPX
    HTTPXClientInstrumentor().instrument()

def get_tracer(name: str):
    """Get tracer instance."""
    return trace.get_tracer(name)
```

### Custom Spans

```python
# app/services/video_service.py
from app.core.tracing import get_tracer

tracer = get_tracer(__name__)

class VideoService:
    async def upload_video(self, file, user, metadata):
        with tracer.start_as_current_span("video_upload") as span:
            span.set_attribute("user_id", str(user.id))
            span.set_attribute("file_size", file.size)
            span.set_attribute("file_type", file.content_type)
            
            # Upload to S3
            with tracer.start_as_current_span("s3_upload") as s3_span:
                upload_result = await self._upload_to_s3(file)
                s3_span.set_attribute("s3_key", upload_result["key"])
                s3_span.set_attribute("s3_bucket", upload_result["bucket"])
            
            # Create database record
            with tracer.start_as_current_span("db_create_video") as db_span:
                video = await self._create_video_record(user, metadata, upload_result)
                db_span.set_attribute("video_id", str(video.id))
            
            # Queue processing job
            with tracer.start_as_current_span("queue_processing") as queue_span:
                job_id = await self._queue_processing_job(video.id)
                queue_span.set_attribute("job_id", job_id)
            
            return {
                "video_id": str(video.id),
                "status": "uploaded",
                "job_id": job_id
            }
```

## 🚨 Alerting Configuration

### AlertManager Rules

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@socialflow.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://pagerduty:9093/webhook'
        send_resolved: true

  - name: 'email'
    email_configs:
      - to: 'alerts@socialflow.com'
        subject: 'Social Flow Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

### Prometheus Alert Rules

```yaml
# monitoring/rules/alerts.yml
groups:
  - name: socialflow-backend
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

      - alert: DatabaseConnectionHigh
        expr: database_connections_active > 80
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High database connection count"
          description: "Database has {{ $value }} active connections"

      - alert: RedisConnectionHigh
        expr: redis_connections_active > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High Redis connection count"
          description: "Redis has {{ $value }} active connections"

      - alert: VideoUploadFailure
        expr: rate(video_uploads_total{status="failed"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High video upload failure rate"
          description: "Video upload failure rate is {{ $value }} per second"

      - alert: LowVideoViews
        expr: rate(video_views_total[1h]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low video view rate"
          description: "Video view rate is {{ $value }} per second"

      - alert: SecurityEvent
        expr: rate(errors_total{error_type=~".*security.*"}[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Security event detected"
          description: "Security error rate is {{ $value }} per second"
```

## 📊 Business Metrics

### Revenue Tracking

```python
# app/core/business_metrics.py
from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any

# Revenue metrics
REVENUE_TOTAL = Counter(
    'revenue_total',
    'Total revenue',
    ['currency', 'source']
)

REVENUE_SUBSCRIPTIONS = Counter(
    'revenue_subscriptions_total',
    'Total subscription revenue',
    ['plan_id', 'currency']
)

REVENUE_ADS = Counter(
    'revenue_ads_total',
    'Total ad revenue',
    ['ad_id', 'currency']
)

REVENUE_CREATOR_PAYOUTS = Counter(
    'revenue_creator_payouts_total',
    'Total creator payouts',
    ['creator_id', 'currency']
)

# User engagement metrics
USER_ENGAGEMENT = Histogram(
    'user_engagement_duration_seconds',
    'User engagement duration',
    ['user_id', 'activity_type']
)

CONTENT_PERFORMANCE = Histogram(
    'content_performance_score',
    'Content performance score',
    ['content_type', 'content_id']
)

# Business KPIs
DAILY_ACTIVE_USERS = Gauge(
    'daily_active_users',
    'Daily active users'
)

MONTHLY_ACTIVE_USERS = Gauge(
    'monthly_active_users',
    'Monthly active users'
)

CONTENT_UPLOAD_RATE = Counter(
    'content_uploads_total',
    'Total content uploads',
    ['content_type', 'user_id']
)

def track_revenue(amount: float, currency: str, source: str, **kwargs: Any) -> None:
    """Track revenue."""
    REVENUE_TOTAL.labels(currency=currency, source=source).inc(amount)
    
    if source == 'subscription':
        plan_id = kwargs.get('plan_id', 'unknown')
        REVENUE_SUBSCRIPTIONS.labels(plan_id=plan_id, currency=currency).inc(amount)
    elif source == 'ads':
        ad_id = kwargs.get('ad_id', 'unknown')
        REVENUE_ADS.labels(ad_id=ad_id, currency=currency).inc(amount)
    elif source == 'creator_payout':
        creator_id = kwargs.get('creator_id', 'unknown')
        REVENUE_CREATOR_PAYOUTS.labels(creator_id=creator_id, currency=currency).inc(amount)

def track_user_engagement(user_id: str, activity_type: str, duration: float) -> None:
    """Track user engagement."""
    USER_ENGAGEMENT.labels(user_id=user_id, activity_type=activity_type).observe(duration)

def track_content_performance(content_type: str, content_id: str, score: float) -> None:
    """Track content performance."""
    CONTENT_PERFORMANCE.labels(content_type=content_type, content_id=content_id).observe(score)

def track_content_upload(content_type: str, user_id: str) -> None:
    """Track content upload."""
    CONTENT_UPLOAD_RATE.labels(content_type=content_type, user_id=user_id).inc()
```

### Analytics Dashboard

```json
{
  "dashboard": {
    "title": "Social Flow Backend - Business Metrics",
    "panels": [
      {
        "title": "Revenue by Source",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(revenue_total[1h])",
            "legendFormat": "{{source}} ({{currency}})"
          }
        ],
        "yAxes": [
          {
            "label": "Revenue/hour"
          }
        ]
      },
      {
        "title": "Daily Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "daily_active_users",
            "legendFormat": "DAU"
          }
        ]
      },
      {
        "title": "Content Upload Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(content_uploads_total[5m])",
            "legendFormat": "{{content_type}}"
          }
        ],
        "yAxes": [
          {
            "label": "Uploads/sec"
          }
        ]
      },
      {
        "title": "User Engagement",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(user_engagement_duration_seconds_bucket[5m]))",
            "legendFormat": "{{activity_type}} (50th percentile)"
          }
        ],
        "yAxes": [
          {
            "label": "Duration (seconds)"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Monitoring

### Security Metrics

```python
# app/core/security_metrics.py
from prometheus_client import Counter, Histogram
from typing import Dict, Any

# Security metrics
FAILED_LOGINS = Counter(
    'failed_logins_total',
    'Total failed login attempts',
    ['username', 'ip_address']
)

SUSPICIOUS_ACTIVITY = Counter(
    'suspicious_activity_total',
    'Total suspicious activity events',
    ['activity_type', 'user_id', 'ip_address']
)

RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint', 'ip_address']
)

SECURITY_EVENTS = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

def track_failed_login(username: str, ip_address: str) -> None:
    """Track failed login attempt."""
    FAILED_LOGINS.labels(username=username, ip_address=ip_address).inc()

def track_suspicious_activity(activity_type: str, user_id: str, ip_address: str) -> None:
    """Track suspicious activity."""
    SUSPICIOUS_ACTIVITY.labels(
        activity_type=activity_type,
        user_id=user_id,
        ip_address=ip_address
    ).inc()

def track_rate_limit_hit(endpoint: str, ip_address: str) -> None:
    """Track rate limit hit."""
    RATE_LIMIT_HITS.labels(endpoint=endpoint, ip_address=ip_address).inc()

def track_security_event(event_type: str, severity: str) -> None:
    """Track security event."""
    SECURITY_EVENTS.labels(event_type=event_type, severity=severity).inc()
```

### Security Dashboard

```json
{
  "dashboard": {
    "title": "Social Flow Backend - Security",
    "panels": [
      {
        "title": "Failed Login Attempts",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(failed_logins_total[5m])",
            "legendFormat": "{{username}} ({{ip_address}})"
          }
        ],
        "yAxes": [
          {
            "label": "Failed logins/sec"
          }
        ]
      },
      {
        "title": "Suspicious Activity",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(suspicious_activity_total[5m])",
            "legendFormat": "{{activity_type}}"
          }
        ],
        "yAxes": [
          {
            "label": "Events/sec"
          }
        ]
      },
      {
        "title": "Rate Limit Hits",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rate_limit_hits_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Hits/sec"
          }
        ]
      },
      {
        "title": "Security Events",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(security_events_total[5m])",
            "legendFormat": "{{event_type}} ({{severity}})"
          }
        ],
        "yAxes": [
          {
            "label": "Events/sec"
          }
        ]
      }
    ]
  }
}
```

## 📱 Health Checks

### Application Health

```python
# app/core/health.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.redis import get_redis
import asyncio
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "socialflow-backend",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis)
):
    """Detailed health check with dependencies."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "socialflow-backend",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database check
    try:
        await db.execute("SELECT 1")
        health_status["checks"]["database"] = {
            "status": "healthy",
            "response_time": 0.001
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Redis check
    try:
        start_time = time.time()
        await redis.ping()
        response_time = time.time() - start_time
        health_status["checks"]["redis"] = {
            "status": "healthy",
            "response_time": response_time
        }
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # S3 check
    try:
        start_time = time.time()
        # Check S3 connectivity
        response_time = time.time() - start_time
        health_status["checks"]["s3"] = {
            "status": "healthy",
            "response_time": response_time
        }
    except Exception as e:
        health_status["checks"]["s3"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@router.get("/health/readiness")
async def readiness_check():
    """Readiness check for Kubernetes."""
    return {"status": "ready"}

@router.get("/health/liveness")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
```

## 🚀 Deployment Monitoring

### Docker Health Checks

```dockerfile
# Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Kubernetes Health Checks

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: socialflow-backend
spec:
  template:
    spec:
      containers:
      - name: socialflow-backend
        image: socialflow-backend:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
```

## 📊 Monitoring Best Practices

### 1. Metric Naming

- Use descriptive names: `http_requests_total` not `requests`
- Include units: `_seconds`, `_bytes`, `_total`
- Use consistent naming conventions

### 2. Label Usage

- Use labels for dimensions that can be aggregated
- Avoid high-cardinality labels
- Keep label values short and meaningful

### 3. Alerting

- Set appropriate thresholds
- Use different severity levels
- Include runbook links in alerts
- Test alerting regularly

### 4. Dashboard Design

- Group related metrics together
- Use appropriate visualization types
- Include time ranges and refresh intervals
- Keep dashboards focused and actionable

### 5. Log Management

- Use structured logging
- Include correlation IDs
- Set appropriate log levels
- Implement log rotation and retention

## 🛠️ Monitoring Tools

### Open Source

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis
- **AlertManager**: Alert routing and management

### Cloud Services

- **AWS CloudWatch**: Metrics, logs, and alarms
- **AWS X-Ray**: Distributed tracing
- **Datadog**: Full-stack monitoring
- **New Relic**: Application performance monitoring
- **Sentry**: Error tracking and performance monitoring

## 📈 Performance Monitoring

### Key Performance Indicators (KPIs)

1. **Response Time**: 95th percentile < 500ms
2. **Throughput**: > 1000 requests/second
3. **Error Rate**: < 0.1%
4. **Availability**: > 99.9%
5. **Database Performance**: Query time < 100ms
6. **Cache Hit Rate**: > 90%

### Performance Testing

```python
# tests/performance/load_test.py
import asyncio
import aiohttp
import time
from typing import List

async def load_test(endpoint: str, concurrent_users: int, duration: int):
    """Run load test against endpoint."""
    async def make_request(session: aiohttp.ClientSession):
        start_time = time.time()
        try:
            async with session.get(endpoint) as response:
                await response.text()
                return {
                    "status": response.status,
                    "duration": time.time() - start_time,
                    "success": response.status < 400
                }
        except Exception as e:
            return {
                "status": 0,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(concurrent_users):
            task = asyncio.create_task(make_request(session))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Calculate metrics
        successful_requests = sum(1 for r in results if r["success"])
        total_requests = len(results)
        success_rate = successful_requests / total_requests
        
        durations = [r["duration"] for r in results]
        avg_duration = sum(durations) / len(durations)
        p95_duration = sorted(durations)[int(len(durations) * 0.95)]
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "p95_duration": p95_duration
        }

# Run load test
if __name__ == "__main__":
    result = asyncio.run(load_test(
        "http://localhost:8000/health",
        concurrent_users=100,
        duration=60
    ))
    print(f"Load test results: {result}")
```

## 🔧 Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks
   - Review garbage collection settings
   - Monitor object creation patterns

2. **Slow Database Queries**
   - Enable query logging
   - Review query execution plans
   - Check for missing indexes

3. **High Error Rates**
   - Check application logs
   - Review error patterns
   - Verify external service dependencies

4. **Performance Degradation**
   - Check resource utilization
   - Review recent deployments
   - Analyze traffic patterns

### Debugging Tools

```python
# app/core/debug.py
import cProfile
import pstats
import io
from functools import wraps

def profile(func):
    """Profile function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Profile for {func.__name__}:")
        print(s.getvalue())
        
        return result
    return wrapper

# Usage
@profile
async def slow_function():
    # Function implementation
    pass
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/)
- [Monitoring Best Practices](https://sre.google/sre-book/monitoring-distributed-systems/)
