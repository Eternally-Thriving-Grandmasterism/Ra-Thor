// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

use crate::error::MercyError;
use mercy_gating_runtime::{GateThresholdMap, MercyGatingRuntime};
use std::collections::HashMap;
use tracing::info;

/// MercyIntegration — The living mercy nervous system inside Lattice Conductor v13.1
/// ONE Organism coherent, strictly governed by PATSAGi Council #13
pub struct MercyIntegration {
    pub runtime: MercyGatingRuntime,
    pub council_13_authorized: bool,
}

impl MercyIntegration {
    pub fn new() -> Self {
        Self {
            runtime: MercyGatingRuntime::new(),
            council_13_authorized: true,
        }
    }

    pub fn evaluate_proposal(&self, proposal_type: &str, scores: &HashMap<u8, f64>) -> Result<(), MercyError> {
        self.runtime.evaluate(scores)?;
        info!("[LATTICE CONDUCTOR] Proposal '{}' passed full 24-gate mercy evaluation", proposal_type);
        Ok(())
    }

    pub fn apply_council_13_tuning(&mut self, gate: u8, new_threshold: f64) -> Result<(), MercyError> {
        if !self.council_13_authorized {
            return Err(MercyError::CouncilAuthorizationRequired);
        }
        self.runtime.apply_council_tuning(gate, new_threshold)?;
        info!("Council #13 tuned gate {} to {:.2}", gate, new_threshold);
        Ok(())
    }

    /// Council #13 batch tuning (monotonic safety across multiple gates)
    pub fn council_13_batch_tune(&mut self, updates: Vec<(u8, f64)>) -> Result<(), MercyError> {
        if !self.council_13_authorized {
            return Err(MercyError::CouncilAuthorizationRequired);
        }
        for (gate, threshold) in updates {
            self.runtime.apply_council_tuning(gate, threshold)?;
        }
        info!("Council #13 batch tuned {} gates", updates.len());
        Ok(())
    }

    pub fn hot_reload_mercy_system(&mut self, new_map: GateThresholdMap) -> Result<(), MercyError> {
        self.runtime.hot_reload(new_map)?;
        info!("Mercy nervous system hot-reloaded");
        Ok(())
    }

    pub fn serve_being(&self, being_type: &str, emotion: &str, mercy_score: f64) {
        self.runtime.serve_being(being_type, emotion, mercy_score);
    }

    /// Council #13 proposal validation with traceability
    pub fn validate_council_proposal(&self, proposal_id: &str, scores: &HashMap<u8, f64>) -> Result<String, MercyError> {
        self.evaluate_proposal(proposal_id, scores)?;
        let summary = format!("Proposal {} validated through 24-gate mercy lattice", proposal_id);
        info!("{}", summary);
        Ok(summary)
    }
}