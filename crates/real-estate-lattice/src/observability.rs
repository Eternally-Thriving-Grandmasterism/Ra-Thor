//! Observability Lattice with Prometheus + OpenTelemetry tracing.

use std::sync::Arc;

// Prometheus (existing)
use prometheus::{Registry, TextEncoder};

// OpenTelemetry
use opentelemetry::global;
use opentelemetry::trace::{Tracer, Span, SpanKind};
use opentelemetry_sdk::trace as sdktrace;
use opentelemetry_otlp::WithExportConfig;

/// Initialize OpenTelemetry tracing (OTLP exporter).
/// Call this once at application startup.
pub fn init_tracing(service_name: &str) -> opentelemetry::sdk::trace::Tracer {
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://localhost:4317"), // OTLP gRPC endpoint
        )
        .with_trace_config(
            sdktrace::config()
                .with_resource(opentelemetry::sdk::Resource::new(vec![opentelemetry::KeyValue::new(
                    "service.name",
                    service_name.to_string(),
                )])),
        )
        .install_simple()
        .expect("Failed to initialize OpenTelemetry");

    global::set_tracer_provider(tracer.clone());
    tracer
}

/// Get a tracer for the valuation system.
pub fn valuation_tracer() -> opentelemetry::sdk::trace::Tracer {
    global::tracer("ra-thor.valuation")
}

// ... existing Telemetry, ValuationObservability, etc. ...

impl ValuationObservability {
    // ... existing methods ...

    /// Record an invalidation with tracing.
    pub fn record_invalidation_with_trace(&self, property_key: &str) {
        let tracer = valuation_tracer();
        let mut span = tracer
            .span_builder("avm_cache_invalidation")
            .with_kind(SpanKind::Internal)
            .start(&tracer);

        span.set_attribute(opentelemetry::KeyValue::new("property_key", property_key.to_string()));

        self.record_invalidation();

        // Span ends automatically when dropped
    }
}
