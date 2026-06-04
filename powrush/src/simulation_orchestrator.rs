//! Reputation-Based Quorum Adjustments for Shard Consensus
//!
//! Dynamically adjusts the required consensus threshold based on the
//! reputation of participating voters. Higher reputation = lower bar.
//! Low reputation participation increases the difficulty of passing proposals.

use crate::simulation_orchestrator::ShardConsensus;

impl ShardConsensus {
    /// Calculate average reputation of voters on a proposal
    pub fn calculate_average_reputation_of_voters(
        &self,
        proposal_id: u64,
        reputation_tracker: &crate::simulation_orchestrator::ShardReputationTracker,
    ) -> f32 {
        if let Some(votes) = self.votes.get(&proposal_id) {
            if votes.is_empty() {
                return 50.0; // default
            }

            let mut total_reputation = 0.0;
            let mut count = 0;

            for vote in votes {
                let rep = reputation_tracker.get_reputation(vote.voter_shard_id);
                total_reputation += rep;
                count += 1;
            }

            return total_reputation / count as f32;
        }
        50.0
    }

    /// Reputation-adjusted consensus check
    pub fn is_consensus_reached_with_reputation(
        &self,
        proposal_id: u64,
        reputation_tracker: &crate::simulation_orchestrator::ShardReputationTracker,
        base_threshold: f32,
    ) -> bool {
        let (approve, reject) = self.tally_votes(proposal_id);

        if !self.has_sufficient_participation(proposal_id) {
            return false;
        }

        let avg_reputation = self.calculate_average_reputation_of_voters(
            proposal_id,
            reputation_tracker,
        );

        // Higher average reputation → lower required threshold
        // Lower average reputation → higher required threshold
        let reputation_factor = (avg_reputation - 50.0) / 100.0; // -0.5 to +0.5
        let adjusted_threshold = (base_threshold - reputation_factor * 0.3).clamp(0.8, 2.5);

        approve > reject && approve >= adjusted_threshold
    }

    /// Minimum reputation required for a vote to count toward quorum
    pub fn get_minimum_reputation_for_quorum(&self) -> f32 {
        20.0 // Votes from shards below 20 reputation have reduced influence
    }
}
