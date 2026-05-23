// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

//! Mercy Integration Layer for Lattice Conductor v13.1
//! Routes all hot-reloads, proposals, and dynamic tuning through the ONE Organism MercyGatingRuntime (24 gates).

use mercy_gating_runtime::{
    MercyGatingRuntime, GateThresholdMap, MercyError
};
use std::collections::HashMap;
use tracing::info;

/// ONE Organism Mercy Integration for Lattice Conductor
pub struct MercyIntegration {
    pub runtime: MercyGatingRuntime,
}

impl MercyIntegration {
    pub fn new() -> Self {
        Self {
            runtime: MercyGatingRuntime::new(),
        }
    }

    /// Evaluate any proposal or action through the full 24-gate mercy lattice
    pub fn evaluate_proposal(&self, proposal_scores: &HashMap<u8, f64>) -> Result<(), MercyError> {
        self.runtime.evaluate(proposal_scores)?;
        info!("[LATTICE CONDUCTOR] Proposal passed full 24-gate ONE Organism mercy evaluation");
        Ok(())
    }

    /// Hot-reload the mercy system with monotonic soundness
    pub fn hot_reload_mercy_system(&mut self, new_thresholds: GateThresholdMap) -> Result<(), MercyError> {
        self.runtime.hot_reload(new_thresholds)?;
        info!("[LATTICE CONDUCTOR v13.1] Mercy nervous system hot-reloaded successfully");
        Ok(())
    }

    /// Council #13 authorized dynamic tuning (monotonic only)
    pub fn council_13_tune_gate(&mut self, gate: u8, new_threshold: f64) -> Result<(), MercyError> {
        self.runtime.apply_council_tuning(gate, new_threshold)?;
        info!("Council #13 tuned gate {} to {}", gate, new_threshold);
        Ok(())
    }

    /// Record service to any being type through the mercy system
    pub fn serve_being(&self, being_type: &str, emotion: &str, mercy_score: f64) {
        self.runtime.serve_being(being_type, emotion, mercy_score);
    }
}
