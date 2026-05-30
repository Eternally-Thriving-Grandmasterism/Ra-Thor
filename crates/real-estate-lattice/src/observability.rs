//! Observability Lattice with Prometheus metrics and HTTP endpoint support.

use std::sync::Arc;

use prometheus::{Registry, TextEncoder};

// ... existing Telemetry, ValuationObservability, etc. ...

/// Helper to generate Prometheus text for HTTP responses.
pub fn metrics_text(registry: &Registry) -> String {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Example integration with Axum (recommended for async Rust).
/// 
/// Add to your Cargo.toml:
/// ```toml
/// [dependencies]
/// axum = { version = "0.7", features = ["macros"] }
/// tokio = { version = "1", features = ["full"] }
/// ```
/// 
/// Then use:
/// ```ignore
/// use axum::{routing::get, Router};
/// use std::sync::Arc;
/// use crate::observability::ValuationObservability;
/// 
/// async fn metrics_handler(observability: Arc<ValuationObservability>) -> String {
///     observability.metrics_text()
/// }
/// 
/// let app = Router::new()
///     .route("/metrics", get({
///         let obs = observability.clone();
///         move || async move { metrics_handler(obs.clone()).await }
///     }));
/// ```
pub mod axum_example {
    // This module contains documentation and example code only.
    // The actual implementation should live in your application binary or API crate.
}
