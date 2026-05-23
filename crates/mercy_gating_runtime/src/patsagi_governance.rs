//! PatsagiGovernance with automatic success recording and decay

use crate::{
    CouncilArbitrationSession, CouncilRegistry, SlashingPolicy, CouncilReputation,
    record_success,
};
#[cfg(feature = "serde")]
use bincode;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PatsagiGovernance {
    pub registry: CouncilRegistry,
    pub policy: SlashingPolicy,
    pub reputations: HashMap<u32, CouncilReputation>,  // Added for reputation tracking
}

impl PatsagiGovernance {
    pub fn new() -> Self {
        Self {
            registry: CouncilRegistry::new(),
            policy: SlashingPolicy::default(),
            reputations: HashMap::new(),
        }
    }

    pub fn with_policy(policy: SlashingPolicy) -> Self {
        Self {
            registry: CouncilRegistry::new(),
            policy,
            reputations: HashMap::new(),
        }
    }

    pub fn run_arbitration(
        &mut self,
        session: &CouncilArbitrationSession,
        current_turn: u64,
    ) -> Vec<crate::CouncilTuningProposal> {
        let accepted = session.accepted_proposals_with_staking(&self.registry.stakes);

        if accepted.is_empty() {
            // slashing logic...
            return vec![];
        }

        for proposal in &accepted {
            self.registry.apply_success_bonus(proposal.council_id, self.policy.success_bonus);

            // Auto record success for reputation
            let rep = self.reputations.entry(proposal.council_id).or_insert(CouncilReputation {
                council_id: proposal.council_id,
                success_count: 0,
                failure_count: 0,
            });
            record_success(rep);
        }

        accepted
    }

    pub fn decay_reputation(&mut self, council_id: u32, decay_amount: u64) {
        if let Some(rep) = self.reputations.get_mut(&council_id) {
            rep.success_count = rep.success_count.saturating_sub(decay_amount);
        }
    }

    // ... serialization methods ...
}