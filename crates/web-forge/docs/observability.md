# Observability Guide for web-forge

This document explains how to set up distributed tracing and metrics for `web-forge` using OpenTelemetry.

## 1. Initialize Observability

```rust
use web_forge::observability::{init_observability, ObservabilityConfig};

init_observability(ObservabilityConfig {
    service_name: "my-service".to_string(),
    otlp_endpoint: Some("http://localhost:4317".to_string()),
    enable_stdout: true,
});
```

## 2. OpenTelemetry Collector (Recommended)

We recommend using the **OpenTelemetry Collector** as the central agent.

### Minimal `otel-collector-config.yaml`

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [logging]
    metrics:
      receivers: [otlp]
      exporters: [prometheus, logging]
```

Run the collector:

```bash
otelcol-contrib --config otel-collector-config.yaml
```

## 3. Prometheus Scraping

Add the following to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'web-forge'
    static_configs:
      - targets: ['localhost:8889']
```

## 4. Metrics Exposed

- `orchestration_duration_seconds` (Histogram)
- `orchestration_success_total` (Counter)
- `orchestration_failure_total` (Counter)

## 5. Viewing Traces

You can send traces to Jaeger, Tempo, or any OTLP-compatible backend by changing the exporter in the Collector.
