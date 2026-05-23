// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0
// See LICENSE-AG-SML for full terms. Zero-harm. Eternal mercy.

//! ONE Organism observability hooks for mercy gating runtime.

pub struct MercyMetrics;

impl MercyMetrics {
    pub fn record_gate_pass(gate: u8, score: f64) {
        tracing::debug!("Gate {} passed with score {:.2}", gate, score);
    }

    pub fn record_threshold_raise(gate: u8, old: f64, new: f64) {
        tracing::info!("Threshold raised for gate {}: {:.2} → {:.2}", gate, old, new);
    }
}