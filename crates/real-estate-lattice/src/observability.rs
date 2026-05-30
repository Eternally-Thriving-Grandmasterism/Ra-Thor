//! Observability Lattice — Reusable, Mercy-Aligned Observability Framework
//!
//! Now includes Prometheus metrics export for production monitoring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts, Registry,
};

/// Core telemetry instruments backed by Prometheus.
pub struct Telemetry {
    pub invalidations_processed: IntCounter,
    pub invalidation_errors: IntCounter,
    pub consumer_restarts: IntCounter,
    pub divergence_events: IntCounter,
    pub registry: Registry,
}

impl Default for Telemetry {
    fn default() -> Self {
        let registry = Registry::new();

        let invalidations_processed = IntCounter::new(
            "ra_thor_invalidation_processed_total",
            "Total number of AVM cache invalidations processed",
        )
        .unwrap();

        let invalidation_errors = IntCounter::new(
            "ra_thor_invalidation_errors_total",
            "Total number of errors during invalidation",
        )
        .unwrap();

        let consumer_restarts = IntCounter::new(
            "ra_thor_consumer_restarts_total",
            "Total number of consumer restarts",
        )
        .unwrap();

        let divergence_events = IntCounter::new(
            "ra_thor_divergence_events_total",
            "Total number of divergence events between external AVM and internal signals",
        )
        .unwrap();

        registry
            .register(Box::new(invalidations_processed.clone()))
            .unwrap();
        registry
            .register(Box::new(invalidation_errors.clone()))
            .unwrap();
        registry
            .register(Box::new(consumer_restarts.clone()))
            .unwrap();
        registry
            .register(Box::new(divergence_events.clone()))
            .unwrap();

        Self {
            invalidations_processed,
            invalidation_errors,
            consumer_restarts,
            divergence_events,
            registry,
        }
    }
}

impl Telemetry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_invalidation(&self) {
        self.invalidations_processed.inc();
    }

    pub fn record_error(&self) {
        self.invalidation_errors.inc();
    }

    pub fn record_consumer_restart(&self) {
        self.consumer_restarts.inc();
    }

    pub fn record_divergence(&self) {
        self.divergence_events.inc();
    }

    /// Returns the Prometheus registry for scraping.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Gather metrics in Prometheus text format.
    pub fn gather(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub invalidations_processed: u64,
    pub invalidation_errors: u64,
    pub consumer_restarts: u64,
    pub divergence_events: u64,
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Critical { reason: String },
}

pub trait ReflectionHook: Send + Sync {
    fn reflect(&self, snapshot: &TelemetrySnapshot) -> Option<String>;
}

pub struct NoopReflection;

impl ReflectionHook for NoopReflection {
    fn reflect(&self, _snapshot: &TelemetrySnapshot) -> Option<String> {
        None
    }
}

pub struct ValuationObservability {
    pub telemetry: Arc<Telemetry>,
    pub reflection: Arc<dyn ReflectionHook>,
}

impl ValuationObservability {
    pub fn new() -> Self {
        Self {
            telemetry: Telemetry::new(),
            reflection: Arc::new(NoopReflection),
        }
    }

    pub fn with_reflection(mut self, hook: Arc<dyn ReflectionHook>) -> Self {
        self.reflection = hook;
        self
    }

    pub fn record_invalidation(&self) {
        self.telemetry.record_invalidation();
    }

    pub fn health_check(&self) -> HealthStatus {
        // Basic health logic (can be expanded)
        HealthStatus::Healthy
    }

    /// Expose Prometheus metrics text.
    pub fn metrics_text(&self) -> String {
        self.telemetry.gather()
    }
}
