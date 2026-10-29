//! # PATSAGi Councils — Powrush Integration Layer
//!
//! This module implements the **13+ PATSAGi Councils** as first-class citizens
//! in the Powrush game world.
//!
//! Each council is a specialized mercy-gated intelligence that influences:
//! - World cycles
//! - Resource regeneration
//! - Player ascension
//! - Faction decisions
//! - Simulation events
//!
//! Councils operate in parallel (like the Ra-Thor lattice) and reach consensus
//! through mercy-weighted voting.

pub mod abundance_council;
pub mod truth_council;
pub mod harmony_council;
pub mod joy_council;
pub mod radical_love_council; // Supreme veto council

use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};

/// Represents a single PATSAGi Council decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDecision {
    pub council_name: String,
    pub approved: bool,
    pub weight: f32,           // 0.0–1.0 influence on final outcome
    pub reason: String,
}

/// Full consensus result from all active councils
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PATSAGiConsensus {
    pub decisions: Vec<CouncilDecision>,
    pub overall_approved: bool,
    pub final_weight: f32,
}

/// Main PATSAGi Council Coordinator for Powrush
pub struct PATSAGiCoordinator {
    // Councils will be instantiated here
}

impl PATSAGiCoordinator {
    pub fn new() -> Self {
        Self {}
    }

    /// Evaluate an action through all relevant councils
    pub async fn evaluate_action(
        &self,
        action_description: &str,
        context: &str,
        mercy_status: &MercyGateStatus,
    ) -> PATSAGiConsensus {
        let mut decisions = Vec::new();

        // Radical Love Council (supreme veto)
        let love_decision = radical_love_council::evaluate(action_description, context);
        decisions.push(love_decision.clone());

        if !love_decision.approved {
            return PATSAGiConsensus {
                decisions,
                overall_approved: false,
                final_weight: 0.0,
            };
        }

        // Other councils (simplified for now — will expand)
        decisions.push(abundance_council::evaluate(action_description, context));
        decisions.push(truth_council::evaluate(action_description, context));
        decisions.push(harmony_council::evaluate(action_description, context));
        decisions.push(joy_council::evaluate(action_description, context));

        let overall_approved = decisions.iter().all(|d| d.approved);
        let final_weight = decisions.iter().map(|d| d.weight).sum::<f32>() / decisions.len() as f32;

        PATSAGiConsensus {
            decisions,
            overall_approved,
            final_weight,
        }
    }
}
