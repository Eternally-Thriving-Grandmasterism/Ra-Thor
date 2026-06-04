//! Shard Consensus Protocol for Powrush
//!
//! Lightweight consensus mechanism for high-impact decisions across shards.
//! Influenced by PATSAGi principles: proposals that increase overall thriving
//! are favored. Uses weighted voting based on ShardTrust.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of consensus proposals
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConsensusTopic {
    AdjustGlobalCouncilInfluence,
    ApproveLargeEntityMigration,
    UpdateHealingFieldParameters,
    ShardTrustAdjustment,
    Custom(String),
}

/// A proposal that shards can vote on
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsensusProposal {
    pub id: u64,
    pub topic: ConsensusTopic,
    pub proposer_shard_id: u64,
    pub description: String,
    pub proposed_value: f32, // Generic value (e.g. new influence level)
    pub timestamp: u64,
}

/// A vote on a proposal
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsensusVote {
    pub proposal_id: u64,
    pub voter_shard_id: u64,
    pub approve: bool,
    pub trust_weight: f32, // The weight of this vote
}

/// Resource that tracks active consensus proposals and votes
#[derive(Resource, Default)]
pub struct ShardConsensus {
    pub active_proposals: HashMap<u64, ConsensusProposal>,
    pub votes: HashMap<u64, Vec<ConsensusVote>>, // proposal_id -> votes
    pub next_proposal_id: u64,
}

impl ShardConsensus {
    pub fn create_proposal(
        &mut self,
        topic: ConsensusTopic,
        proposer_shard_id: u64,
        description: String,
        proposed_value: f32,
    ) -> u64 {
        let id = self.next_proposal_id;
        self.next_proposal_id += 1;

        let proposal = ConsensusProposal {
            id,
            topic,
            proposer_shard_id,
            description,
            proposed_value,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.active_proposals.insert(id, proposal);
        self.votes.insert(id, vec![]);

        id
    }

    pub fn cast_vote(&mut self, vote: ConsensusVote) {
        if let Some(votes) = self.votes.get_mut(&vote.proposal_id) {
            votes.push(vote);
        }
    }

    /// Tally votes with trust weighting. Returns (approve_weight, reject_weight)
    pub fn tally_votes(&self, proposal_id: u64) -> (f32, f32) {
        let mut approve = 0.0;
        let mut reject = 0.0;

        if let Some(votes) = self.votes.get(&proposal_id) {
            for vote in votes {
                if vote.approve {
                    approve += vote.trust_weight;
                } else {
                    reject += vote.trust_weight;
                }
            }
        }

        (approve, reject)
    }

    pub fn is_consensus_reached(&self, proposal_id: u64, threshold: f32) -> bool {
        let (approve, reject) = self.tally_votes(proposal_id);
        approve > threshold && approve > reject
    }
}

/// Event to propose something for shard consensus
#[derive(Event)]
pub struct ProposeConsensus {
    pub topic: ConsensusTopic,
    pub description: String,
    pub proposed_value: f32,
}

/// System that processes consensus proposals and votes coming from other shards
pub fn process_consensus_messages(
    mut consensus: ResMut<ShardConsensus>,
    mut communicator: ResMut<crate::simulation_orchestrator::CrossShardCommunicator>,
) {
    // In a real implementation, this would deserialize CrossShardMessage::ConsensusProposal / Vote
    // and call create_proposal / cast_vote accordingly.
    // For now this is a placeholder showing the architecture.
}
