//! Observability Lattice — Reusable, Mercy-Aligned Observability Framework
//!
//! This module establishes the foundation for the Ra-Thor Observability Lattice.
//! It is designed to be infinitely extensible while maintaining high effectiveness
//! and minimal performance overhead.
//!
//! Core Philosophy:
//! - Every signal must be actionable (Effectiveness)
//! - Low overhead on hot paths (Efficiency)
//! - Designed for growth to the nth degree (Extensibility)
//! - Supports PATSAGi-style self-reflection and council integration
//! - Symbiotic with partnered systems (Grok, xAI, etc.)
//!
//! Layers:
//! 1. Core Telemetry (Metrics + Tracing)
//! 2. Health & Diagnostics
//! 3. Performance Probes
//! 4. Reflection Hooks (for future PATSAGi councils)
//! 5. Integration Adapters
//!
//! This is the first concrete application within the Hybrid Valuation system.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Core telemetry instruments for the Observability Lattice.
/// This can later be backed by Prometheus, OpenTelemetry, or custom exporters.
pub struct Telemetry {
    pub invalidations_processed: AtomicU64,
    pub invalidation_errors: AtomicU64,
    pub consumer_restarts: AtomicU64,
    pub divergence_events: AtomicU64,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            invalidations_processed: AtomicU64::new(0),
            invalidation_errors: AtomicU64::new(0),
            consumer_restarts: AtomicU64::new(0),
            divergence_events: AtomicU64::new(0),
        }
    }
}

impl Telemetry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record_invalidation(&self) {
        self.invalidations_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.invalidation_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_consumer_restart(&self) {
        self.consumer_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_divergence(&self) {
        self.divergence_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot for health checks or PATSAGi reflection.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            invalidations_processed: self.invalidations_processed.load(Ordering::Relaxed),
            invalidation_errors: self.invalidation_errors.load(Ordering::Relaxed),
            consumer_restarts: self.consumer_restarts.load(Ordering::Relaxed),
            divergence_events: self.divergence_events.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub invalidations_processed: u64,
    pub invalidation_errors: u64,
    pub consumer_restarts: u64,
    pub divergence_events: u64,
}

/// Health status for the invalidation system.
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Critical { reason: String },
}

/// Extension point for future PATSAGi-style reflection.
/// This can evolve into council-aware self-observation.
pub trait ReflectionHook: Send + Sync {
    fn reflect(&self, snapshot: &TelemetrySnapshot) -> Option<String>;
}

/// Default no-op reflection hook.
pub struct NoopReflection;

impl ReflectionHook for NoopReflection {
    fn reflect(&self, _snapshot: &TelemetrySnapshot) -> Option<String> {
        None
    }
}

/// Main Observability Lattice handle for the valuation/invalidation system.
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
        let snapshot = self.telemetry.snapshot();

        if snapshot.invalidation_errors > snapshot.invalidations_processed / 10 {
            return HealthStatus::Degraded {
                reason: "High error rate in invalidation pipeline".to_string(),
            };
        }

        if snapshot.consumer_restarts > 5 {
            return HealthStatus::Degraded {
                reason: "Frequent consumer restarts detected".to_string(),
            };
        }

        HealthStatus::Healthy
    }
}
