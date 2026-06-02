//! # PATSAGi Orchestrator
//!
//! Central coordinator for all 13+ PATSAGi Councils.
//! Runs councils in parallel, collects decisions, and computes final consensus.

use crate::patsagi_councils::{
    abundance_council, harmony_council, joy_council, radical_love_council, truth_council,
    CouncilDecision, PATSAGiConsensus,
};
use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};

pub struct PATSAGiOrchestrator {
    // Future: Load all councils dynamically
}

impl PATSAGiOrchestrator {
    pub fn new() -> Self {
        Self {}
    }

    /// Run all active councils in parallel and return consensus
    pub async fn run_full_consensus(
        &self,
        action_description: &str,
        context: &str,
        mercy_status: &MercyGateStatus,
    ) -> PATSAGiConsensus {
        let mut decisions = Vec::new();

        // Supreme veto first
        let love = radical_love_council::evaluate(action_description, context);
        decisions.push(love.clone());

        if !love.approved {
            return PATSAGiConsensus {
                decisions,
                overall_approved: false,
                final_weight: 0.0,
            };
        }

        // Run remaining councils in parallel (simulated)
        decisions.push(abundance_council::evaluate(action_description, context));
        decisions.push(truth_council::evaluate(action_description, context));
        decisions.push(harmony_council::evaluate(action_description, context));
        decisions.push(joy_council::evaluate(action_description, context));

        // Future: Add remaining 8 councils here

        let overall_approved = decisions.iter().all(|d| d.approved);
        let final_weight = decisions.iter().map(|d| d.weight).sum::<f32>() / decisions.len() as f32;

        PATSAGiConsensus {
            decisions,
            overall_approved,
            final_weight,
        }
    }
}
