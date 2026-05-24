/// Observability Module
///
/// Production-grade tracing and OpenTelemetry integration for web-forge.
///
/// This module provides clean initialization and helpers for distributed tracing
/// and metrics, designed to integrate tightly with `AdvancedOrchestrator`.

use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::{runtime::Tokio, trace as sdktrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Configuration for observability initialization.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub service_name: String,
    pub otlp_endpoint: Option<String>,
    pub enable_stdout: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            service_name: "web-forge".to_string(),
            otlp_endpoint: None,
            enable_stdout: true,
        }
    }
}

/// Initialize tracing and OpenTelemetry.
///
/// This should be called once at application startup.
pub fn init_observability(config: ObservabilityConfig) {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let mut layers = Vec::new();

    // OTLP Exporter (recommended for production)
    if let Some(endpoint) = &config.otlp_endpoint {
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint),
            )
            .with_trace_config(
                sdktrace::config().with_resource(opentelemetry_sdk::Resource::new(vec![
                    KeyValue::new("service.name", config.service_name.clone()),
                ])),
            )
            .install_batch(Tokio)
            .expect("Failed to install OpenTelemetry tracer");

        let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        layers.push(telemetry_layer.boxed());
    }

    // Stdout layer for development
    if config.enable_stdout {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(true);
        layers.push(fmt_layer.boxed());
    }

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(layers)
        .init();

    tracing::info!("Observability initialized (service={})", config.service_name);
}

/// Create a span for an orchestration phase.
#[macro_export]
macro_rules! orchestration_span {
    ($name:expr, $($field:tt)*) => {
        tracing::info_span!(
            concat!("orchestration.", $name),
            $($field)*
        )
    };
}

/// Helper to enter a span for a specific orchestration phase.
pub fn enter_phase_span(phase: &str) -> tracing::span::EnteredSpan {
    tracing::info_span!("orchestration.phase", phase = phase).entered()
}
