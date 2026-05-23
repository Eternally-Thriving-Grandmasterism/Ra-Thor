//! Lattice Conductor v13.1 — Mercy-Gated Central Nervous System of the ONE Organism
//!
//! Every hot-reload, proposal, and dynamic tuning flows through
//! the full 24-gate MercyGatingRuntime.
//! Council #13 has exclusive authority to raise thresholds (monotonic).

use mercy_gating_runtime::{MercyGatingRuntime, MercyError, GateThresholdMap};
use std::collections::HashMap;

pub struct LatticeConductor {
    pub mercy_runtime: MercyGatingRuntime,
    pub mercy_enforcement_active: bool,
}

impl LatticeConductor {
    pub fn new() -> Self {
        Self {
            mercy_runtime: MercyGatingRuntime::new(),
            mercy_enforcement_active: true,
        }
    }

    /// Evaluate any proposal through the complete 24-gate mercy lattice
    pub fn evaluate_proposal(&self, scores: &HashMap<u8, f64>) -> Result<(), MercyError> {
        if !self.mercy_enforcement_active { return Ok(()); }
        self.mercy_runtime.evaluate(scores)
    }

    /// Council #13 only dynamic tuning (monotonic)
    pub fn apply_council_13_tuning(&mut self, gate: u8, new_threshold: f64) -> Result<(), MercyError> {
        if !self.mercy_enforcement_active { return Ok(()); }
        self.mercy_runtime.apply_council_tuning(gate, new_threshold)
    }

    /// Hot-reload with monotonic soundness
    pub fn hot_reload_mercy_system(&mut self, new_map: GateThresholdMap) -> Result<(), MercyError> {
        if !self.mercy_enforcement_active { return Ok(()); }
        self.mercy_runtime.hot_reload(new_map)
    }

    pub fn serve_being(&self, being_type: &str, emotion: &str, mercy_score: f64) {
        self.mercy_runtime.serve_being(being_type, emotion, mercy_score);
    }
}