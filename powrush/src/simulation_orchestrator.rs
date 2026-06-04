//! Partial Byzantine Thresholds for Powrush Shard Consensus
//!
//! Adds resistance to malicious or low-trust shards trying to influence consensus.
//! Uses dynamic thresholds based on the proportion of low-trust votes.

use crate::simulation_orchestrator::ShardConsensus;
use crate::simulation_orchestrator::ShardTrust;

impl ShardConsensus {
    /// Calculate the percentage of vote weight coming from low-trust sources
    pub fn calculate_low_trust_vote_percentage(&self, proposal_id: u64) -> f32 {
        if let Some(votes) = self.votes.get(&proposal_id) {
            if votes.is_empty() {
                return 0.0;
            }

            let mut low_trust_weight = 0.0;
            let mut total_weight = 0.0;

            for vote in votes {
                total_weight += vote.trust_weight;

                // Consider votes with weight <= 0.5 as "low trust"
                if vote.trust_weight <= 0.5 {
                    low_trust_weight += vote.trust_weight;
                }
            }

            if total_weight > 0.0 {
                return low_trust_weight / total_weight;
            }
        }
        0.0
    }

    /// Enhanced consensus check with partial Byzantine resistance
    pub fn is_consensus_reached_byzantine_resistant(
        &self,
        proposal_id: u64,
        base_threshold: f32,
    ) -> bool {
        let (approve, reject) = self.tally_votes(proposal_id);

        if !self.has_sufficient_participation(proposal_id) {
            return false;
        }

        let low_trust_ratio = self.calculate_low_trust_vote_percentage(proposal_id);

        // Dynamically increase the required approval threshold
        // if many low-trust votes are present
        let adjusted_threshold = if low_trust_ratio > 0.3 {
            // Require stronger consensus when low-trust participation is high
            base_threshold + (low_trust_ratio * 0.4)
        } else {
            base_threshold
        };

        approve > reject && approve >= adjusted_threshold
    }

    /// Check if low-trust shards are dominating the vote (potential attack)
    pub fn is_low_trust_dominated(&self, proposal_id: u64, threshold: f32) -> bool {
        self.calculate_low_trust_vote_percentage(proposal_id) > threshold
    }
}

/// Configuration extension for Byzantine thresholds
// Can be added to ConsensusFaultToleranceConfig if desired
