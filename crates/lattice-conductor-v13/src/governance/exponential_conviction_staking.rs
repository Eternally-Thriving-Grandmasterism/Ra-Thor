//! # Enhanced Exponential Conviction Staking
//!
//! Thunder Lattice Governance Primitive (v14 Phase 14.1)
//!
//! Time + mercy-weighted conviction staking.
//! Influence compounds exponentially when aligned with mercy outcomes.
//!
//! This module is designed to be:
//! - Explicit and auditable
//! - Fully integrated with Mercy Gating Runtime
//! - Compatible with Self-Evolution Looping Systems
//! - Testable in isolation and in full governance cycles

use serde::{Deserialize, Serialize};

/// Represents a participant's conviction stake on a proposal or decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvictionStake {
    pub participant_id: String,
    pub proposal_id: String,
    pub conviction_amount: f64,
    pub mercy_alignment_score: f64, // 0.0 – 2.0+
    pub stake_timestamp: u64,       // Unix timestamp or logical epoch
    pub last_update_epoch: u64,
}

impl ConvictionStake {
    pub fn new(
        participant_id: &str,
        proposal_id: &str,
        initial_conviction: f64,
        mercy_alignment: f64,
        current_epoch: u64,
    ) -> Self {
        Self {
            participant_id: participant_id.to_string(),
            proposal_id: proposal_id.to_string(),
            conviction_amount: initial_conviction.max(0.0),
            mercy_alignment_score: mercy_alignment.clamp(0.0, 2.0),
            stake_timestamp: current_epoch,
            last_update_epoch: current_epoch,
        }
    }

    /// Calculates current conviction with exponential time + mercy weighting.
    /// This is the core of Enhanced Exponential Conviction Staking.
    pub fn current_conviction(&self, current_epoch: u64) -> f64 {
        if self.conviction_amount <= 0.0 {
            return 0.0;
        }

        let time_delta = (current_epoch.saturating_sub(self.last_update_epoch)) as f64;

        // Base exponential growth (time-weighted)
        let time_factor = (1.0 + 0.015 * time_delta).powf(1.8);

        // Mercy alignment provides gentle amplification
        let mercy_multiplier = 0.85 + (self.mercy_alignment_score * 0.35);

        (self.conviction_amount * time_factor * mercy_multiplier).max(0.0)
    }

    /// Applies mercy alignment decay when behavior falls out of alignment.
    pub fn apply_decay(&mut self, decay_factor: f64, current_epoch: u64) {
        self.conviction_amount *= decay_factor.clamp(0.6, 1.0);
        self.last_update_epoch = current_epoch;
    }

    /// Updates mercy alignment score (called when new evidence arrives).
    pub fn update_mercy_alignment(&mut self, new_score: f64, current_epoch: u64) {
        self.mercy_alignment_score = new_score.clamp(0.0, 2.0);
        self.last_update_epoch = current_epoch;
    }
}

/// Calculates relative influence ordering for multiple stakes on the same proposal.
pub fn calculate_relative_influence(stakes: &[ConvictionStake], current_epoch: u64) -> Vec<(String, f64)> {
    let mut results: Vec<(String, f64)> = stakes
        .iter()
        .map(|s| (s.participant_id.clone(), s.current_conviction(current_epoch)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_growth_with_high_mercy() {
        let stake = ConvictionStake::new("p1", "prop-42", 10.0, 1.4, 1000);
        let conviction_at_t0 = stake.current_conviction(1000);
        let conviction_later = stake.current_conviction(1100);

        assert!(conviction_later > conviction_at_t0);
    }

    #[test]
    fn test_mercy_alignment_affects_growth() {
        let high_mercy = ConvictionStake::new("p1", "prop-1", 10.0, 1.5, 1000);
        let low_mercy = ConvictionStake::new("p2", "prop-1", 10.0, 0.6, 1000);

        let high_later = high_mercy.current_conviction(1200);
        let low_later = low_mercy.current_conviction(1200);

        assert!(high_later > low_later);
    }
}