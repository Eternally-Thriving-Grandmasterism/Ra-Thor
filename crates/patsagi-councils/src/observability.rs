//! crates/patsagi-councils/src/observability.rs — v14.15.5
//! Lightweight observability for the dual-repo soft feedback organism
//!
//! Tracks the breath of the closed loop:
//!   - Hints emitted
//!   - Valence outcomes (approved / progressive / mercy-block)
//!   - Emission success vs blocked
//!   - Simple running averages
//!
//! Zero heavy dependencies. Snapshot-friendly. Offline-first.
//! Contact: info@Rathor.ai
//! TOLC 8 | Living Cosmic Tick | ONE Organism

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Aggregate metrics for the soft feedback + valence organism.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResonanceMetrics {
    // Cycle counts
    pub total_cycles: u64,
    pub emission_successes: u64,
    pub emission_blocks: u64,
    pub mercy_gate_failures: u64,
    pub no_hints_count: u64,

    // Valence outcomes
    pub valence_approvals: u64,
    pub valence_progressive: u64,
    pub valence_mercy_blocks: u64,
    pub valence_reviews: u64,

    // Hint volume
    pub total_hints_emitted: u64,

    // Running sums for averages (divide by total_cycles or valence decisions)
    pub sum_composite_valence: f64,
    pub sum_joy: f64,
    pub valence_decisions: u64,

    // Timestamps
    pub last_emission_unix: Option<u64>,
    pub last_cycle_unix: Option<u64>,
}

impl ResonanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Record a successful emission cycle.
    pub fn record_emission_success(&mut self, hints: usize) {
        self.total_cycles = self.total_cycles.saturating_add(1);
        self.emission_successes = self.emission_successes.saturating_add(1);
        self.total_hints_emitted = self.total_hints_emitted.saturating_add(hints as u64);
        let now = Self::now_unix();
        self.last_emission_unix = Some(now);
        self.last_cycle_unix = Some(now);
    }

    /// Record a blocked / failed emission attempt.
    pub fn record_emission_block(&mut self, reason: BlockReason) {
        self.total_cycles = self.total_cycles.saturating_add(1);
        self.emission_blocks = self.emission_blocks.saturating_add(1);
        self.last_cycle_unix = Some(Self::now_unix());

        match reason {
            BlockReason::MercyGate => {
                self.mercy_gate_failures = self.mercy_gate_failures.saturating_add(1);
            }
            BlockReason::NoHints => {
                self.no_hints_count = self.no_hints_count.saturating_add(1);
            }
            BlockReason::Other => {}
        }
    }

    /// Record a valence deliberation outcome.
    pub fn record_valence(
        &mut self,
        approved: bool,
        progressive: bool,
        mercy_block: bool,
        composite: f64,
        joy: f64,
    ) {
        self.valence_decisions = self.valence_decisions.saturating_add(1);
        self.sum_composite_valence += composite;
        self.sum_joy += joy;

        if approved {
            self.valence_approvals = self.valence_approvals.saturating_add(1);
        } else if progressive {
            self.valence_progressive = self.valence_progressive.saturating_add(1);
        } else if mercy_block {
            self.valence_mercy_blocks = self.valence_mercy_blocks.saturating_add(1);
        } else {
            self.valence_reviews = self.valence_reviews.saturating_add(1);
        }
    }

    pub fn avg_composite_valence(&self) -> f64 {
        if self.valence_decisions == 0 {
            0.0
        } else {
            self.sum_composite_valence / self.valence_decisions as f64
        }
    }

    pub fn avg_joy(&self) -> f64 {
        if self.valence_decisions == 0 {
            0.0
        } else {
            self.sum_joy / self.valence_decisions as f64
        }
    }

    pub fn emission_success_rate(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            self.emission_successes as f64 / self.total_cycles as f64
        }
    }

    pub fn progressive_rate(&self) -> f64 {
        if self.valence_decisions == 0 {
            0.0
        } else {
            self.valence_progressive as f64 / self.valence_decisions as f64
        }
    }

    pub fn mercy_block_rate(&self) -> f64 {
        if self.valence_decisions == 0 {
            0.0
        } else {
            self.valence_mercy_blocks as f64 / self.valence_decisions as f64
        }
    }

    /// Human-readable snapshot for logs / dashboards.
    pub fn summary(&self) -> String {
        format!(
            "ResonanceMetrics | cycles={} | emit_ok={} ({:.1}%) | hints={} | \
             valence: approve={} progressive={} mercy_block={} review={} | \
             avg_valence={:.3} avg_joy={:.3}",
            self.total_cycles,
            self.emission_successes,
            self.emission_success_rate() * 100.0,
            self.total_hints_emitted,
            self.valence_approvals,
            self.valence_progressive,
            self.valence_mercy_blocks,
            self.valence_reviews,
            self.avg_composite_valence(),
            self.avg_joy()
        )
    }

    /// Reset all counters (useful between test runs).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BlockReason {
    MercyGate,
    NoHints,
    Other,
}

/// Simple shared handle for higher layers (bridge, tests, host).
#[derive(Debug, Clone, Default)]
pub struct MetricsHandle {
    pub inner: ResonanceMetrics,
}

impl MetricsHandle {
    pub fn new() -> Self {
        Self {
            inner: ResonanceMetrics::new(),
        }
    }

    pub fn metrics(&self) -> &ResonanceMetrics {
        &self.inner
    }

    pub fn metrics_mut(&mut self) -> &mut ResonanceMetrics {
        &mut self.inner
    }

    pub fn summary(&self) -> String {
        self.inner.summary()
    }
}
