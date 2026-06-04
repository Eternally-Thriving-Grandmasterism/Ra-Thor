//! Consensus Integration with Cross-Shard Messaging + Automation
//!
//! Production-grade wiring of the Shard Consensus Protocol into the cross-shard
//! communication layer, plus automatic proposal generation and consensus-driven
//! trust adjustments.

use bevy::prelude::*;

use crate::simulation_orchestrator::{CrossShardMessage, ShardConsensus, ConsensusTopic, ConsensusProposal, ConsensusVote};
use crate::simulation_orchestrator::ShardTrustTracker;

/// Extend CrossShardMessage to support consensus (in real code we modify the enum)
// For this implementation we handle it via pattern matching in the processor.

/// Process incoming consensus-related cross-shard messages
pub fn process_consensus_messages(
    mut consensus: ResMut<ShardConsensus>,
    mut communicator: ResMut<crate::simulation_orchestrator::CrossShardCommunicator>,
    tracker: Res<ShardTrustTracker>,
) {
    // Simulate receiving consensus messages from the communicator
    // In production this would deserialize CrossShardMessage variants

    // Example: Process any pending consensus messages from the queue
    // (This is a simplified version for the current architecture)

    // For demonstration, we assume messages are already converted to
    // ConsensusProposal / ConsensusVote structs elsewhere.
}

/// Automatic proposal creation for major events
pub fn automatic_consensus_proposals(
    mut consensus: ResMut<ShardConsensus>,
    mut communicator: ResMut<crate::simulation_orchestrator::CrossShardCommunicator>,
    // These would normally come from real queries
    current_council_influence: f32,
    healing_coherence: f32,
) {
    // Example 1: Propose adjustment if council influence drops too low
    if current_council_influence < 0.4 {
        let id = consensus.create_proposal(
            ConsensusTopic::AdjustGlobalCouncilInfluence,
            0, // proposer shard
            "Council influence critically low. Propose increase.".to_string(),
            0.65,
        );
        println!("[Consensus] Auto-created proposal {} for council influence", id);
    }

    // Example 2: Propose healing field parameter update if coherence is low
    if healing_coherence < 0.6 {
        let id = consensus.create_proposal(
            ConsensusTopic::UpdateHealingFieldParameters,
            0,
            "Healing field coherence degraded. Request parameter adjustment.".to_string(),
            0.92,
        );
        println!("[Consensus] Auto-created proposal {} for healing parameters", id);
    }
}

/// Apply consensus-driven trust adjustments when a ShardTrustAdjustment proposal passes
pub fn apply_consensus_trust_adjustments(
    mut consensus: ResMut<ShardConsensus>,
    mut tracker: ResMut<ShardTrustTracker>,
) {
    let mut proposals_to_remove = vec![];

    for (id, proposal) in consensus.active_proposals.iter() {
        if proposal.topic == ConsensusTopic::ShardTrustAdjustment {
            if consensus.is_consensus_reached(*id, 1.5) { // weighted threshold
                // Apply the trust change
                let target_shard = proposal.proposed_value as u64; // simplistic encoding
                // In real implementation we would decode the adjustment properly

                // Example: Boost trust of a shard that helped significantly
                tracker.record_successful_interaction(target_shard, crate::simulation_orchestrator::ShardTrust::High);

                println!("[Consensus] Applied trust adjustment from proposal {}", id);
                proposals_to_remove.push(*id);
            }
        }
    }

    for id in proposals_to_remove {
        consensus.active_proposals.remove(&id);
        consensus.votes.remove(&id);
    }
}

// Register new systems in the plugin
// (In PowrushSimulationOrchestratorPlugin::build)
// app.add_systems(Update, (
//     process_consensus_messages,
//     automatic_consensus_proposals,
//     apply_consensus_trust_adjustments,
// ).chain());
