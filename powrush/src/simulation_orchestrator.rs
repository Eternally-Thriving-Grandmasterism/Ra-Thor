//! Consensus Fault Tolerance for Powrush Shard Consensus
//!
//! Adds robustness against crashed, slow, or malicious shards.
//! Uses trust-weighted quorums and proposal expiration rather than full BFT
//! (appropriate for a mercy-aligned, thriving-focused simulation).

use bevy::prelude::*;
use std::collections::HashMap;

use crate::simulation_orchestrator::ShardConsensus;

/// Configuration for fault-tolerant consensus
#[derive(Resource)]
pub struct ConsensusFaultToleranceConfig {
    pub proposal_timeout_seconds: u64,
    pub minimum_participation_weight: f32, // e.g. 0.6 = 60% of known trust must vote
    pub max_proposals_per_shard: u32,
}

impl Default for ConsensusFaultToleranceConfig {
    fn default() -> Self {
        Self {
            proposal_timeout_seconds: 300, // 5 minutes
            minimum_participation_weight: 0.6,
            max_proposals_per_shard: 3,
        }
    }
}

impl ShardConsensus {
    /// Check if a proposal has expired
    pub fn is_proposal_expired(&self, proposal_id: u64, current_time: u64, timeout: u64) -> bool {
        if let Some(proposal) = self.active_proposals.get(&proposal_id) {
            return current_time > proposal.timestamp + timeout;
        }
        true
    }

    /// Calculate total trust weight of all known shards (simplified)
    pub fn estimate_total_trust_weight(&self) -> f32 {
        // In a real system this would come from ShardTrustTracker
        // For now we use a placeholder
        10.0
    }

    /// Check if enough weighted votes have been cast (crash fault tolerance)
    pub fn has_sufficient_participation(&self, proposal_id: u64) -> bool {
        let (approve, reject) = self.tally_votes(proposal_id);
        let total_voted = approve + reject;
        let estimated_total = self.estimate_total_trust_weight();

        total_voted >= estimated_total * 0.6 // 60% participation threshold
    }

    /// Enhanced consensus check with fault tolerance
    pub fn is_consensus_reached_fault_tolerant(&self, proposal_id: u64, timeout: u64, current_time: u64) -> bool {
        if self.is_proposal_expired(proposal_id, current_time, timeout) {
            return false;
        }

        if !self.has_sufficient_participation(proposal_id) {
            return false;
        }

        let (approve, reject) = self.tally_votes(proposal_id);
        approve > reject && approve > 1.2 // weighted threshold
    }

    /// Clean up expired proposals
    pub fn cleanup_expired_proposals(&mut self, current_time: u64, timeout: u64) {
        let mut to_remove = vec![];

        for (id, proposal) in self.active_proposals.iter() {
            if current_time > proposal.timestamp + timeout {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.active_proposals.remove(&id);
            self.votes.remove(&id);
        }
    }
}

/// System that maintains consensus fault tolerance (timeouts + cleanup)
pub fn consensus_fault_tolerance_maintenance(
    mut consensus: ResMut<ShardConsensus>,
    config: Res<crate::simulation_orchestrator::ConsensusFaultToleranceConfig>,
) {
    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    consensus.cleanup_expired_proposals(current_time, config.proposal_timeout_seconds);
}
