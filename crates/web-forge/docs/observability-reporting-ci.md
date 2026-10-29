# Observability, Reporting & CI Integration

This document explains how to use `web-forge`'s observability, automated reporting, and CI integration features.

## 1. Observability

`web-forge` provides built-in support for distributed tracing and metrics using OpenTelemetry.

### Initialization

```rust
use web_forge::observability::{init_observability, ObservabilityConfig};

init_observability(ObservabilityConfig {
    service_name: "my-service".to_string(),
    otlp_endpoint: Some("http://localhost:4317".to_string()),
    enable_stdout: true,
});
```

### Recommended Stack

- **OpenTelemetry Collector** (recommended central agent)
- **Prometheus** for metrics
- **Jaeger / Tempo** for traces
- **Grafana** for dashboards

## 2. Automated Reporting

`OrchestrationReport` provides structured output from orchestration runs.

```rust
use web_forge::orchestration::OrchestrationReport;

let report = OrchestrationReport::from(&result);
println!("{}", report.summary());

// JSON for CI artifacts
let json = report.to_json();
```

### CI Quality Gate

```rust
if report.should_fail_ci(80.0) {
    eprintln!("Quality gate failed");
    std::process::exit(1);
}
```

## 3. GitHub Actions Integration

A ready-to-use workflow is available at:
`.github/workflows/web-forge-ci.yml`

### Features

- Matrix testing (Ubuntu, Windows, macOS + stable/beta)
- Rust caching
- Parallel job execution
- Quality gates based on success + WCAG AA score
- Artifact upload for reports

### Recommended Quality Gate

Fail the pipeline if:
- Orchestration was not successful, **or**
- WCAG AA score is below your defined threshold (e.g. 75)

## 4. Best Practices

- Always initialize observability early in your application
- Use `should_fail_ci()` for clear CI decision making
- Upload reports as artifacts for traceability
- Combine tracing + metrics for full observability
