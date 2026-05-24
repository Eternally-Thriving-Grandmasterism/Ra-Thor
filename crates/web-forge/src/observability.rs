/// Observability Module
///
/// Production-grade tracing + OpenTelemetry metrics for web-forge.

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram};
use std::time::Instant;
use tracing::info_span;

// Lazy static metrics (initialized on first use)
static ORCHESTRATION_DURATION: once_cell::sync::Lazy<Histogram<f64>> =
    once_cell::sync::Lazy::new(|| {
        global::meter("web_forge")
            .f64_histogram("orchestration_duration_seconds")
            .with_description("Duration of orchestration runs in seconds")
            .init()
    });

static ORCHESTRATION_SUCCESS: once_cell::sync::Lazy<Counter<u64>> =
    once_cell::sync::Lazy::new(|| {
        global::meter("web_forge")
            .u64_counter("orchestration_success_total")
            .with_description("Total successful orchestrations")
            .init()
    });

static ORCHESTRATION_FAILURE: once_cell::sync::Lazy<Counter<u64>> =
    once_cell::sync::Lazy::new(|| {
        global::meter("web_forge")
            .u64_counter("orchestration_failure_total")
            .with_description("Total failed orchestrations")
            .init()
    });

/// Records metrics for a completed orchestration run.
pub fn record_orchestration_metrics(duration_secs: f64, success: bool, attempts: usize) {
    ORCHESTRATION_DURATION.record(duration_secs, &[]);

    if success {
        ORCHESTRATION_SUCCESS.add(1, &[]);
    } else {
        ORCHESTRATION_FAILURE.add(1, &[]);
    }

    tracing::info!(
        duration_secs = duration_secs,
        success = success,
        attempts = attempts,
        "Orchestration metrics recorded"
    );
}

/// Simple helper to time an orchestration block.
pub fn time_orchestration<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_secs_f64();
    (result, duration)
}
