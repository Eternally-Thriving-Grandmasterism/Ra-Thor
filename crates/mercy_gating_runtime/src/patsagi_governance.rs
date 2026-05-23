//! PatsagiGovernance with real serialization support

use crate::{
    CouncilArbitrationSession, CouncilRegistry, SlashingPolicy,
    calculate_slash_amount, apply_slash,
};
#[cfg(feature = "serde")]
use bincode;

#[derive(Debug, Clone)]
pub struct PatsagiGovernance {
    pub registry: CouncilRegistry,
    pub policy: SlashingPolicy,
}

impl PatsagiGovernance {
    pub fn new() -> Self {
        Self {
            registry: CouncilRegistry::new(),
            policy: SlashingPolicy::default(),
        }
    }

    pub fn run_arbitration(
        &mut self,
        session: &CouncilArbitrationSession,
        current_turn: u64,
    ) -> Vec<crate::CouncilTuningProposal> {
        let accepted = session.accepted_proposals_with_staking(&self.registry.stakes);

        if accepted.is_empty() {
            for proposal in &session.proposals {
                if let Some(stake) = self.registry.get_stake_mut(proposal.council_id) {
                    if !stake.can_propose(&proposal.target, current_turn) {
                        let slash = calculate_slash_amount(&self.policy, proposal, 0);
                        apply_slash(stake, slash);
                    }
                }
            }
            return vec![];
        }

        for proposal in &accepted {
            self.registry.apply_success_bonus(proposal.council_id, self.policy.success_bonus);
        }

        accepted
    }

    #[cfg(feature = "serde")]
    pub fn serialize_registry(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(&self.registry)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    #[cfg(feature = "serde")]
    pub fn deserialize_registry(&mut self, data: &[u8]) -> Result<(), String> {
        self.registry = bincode::deserialize(data)
            .map_err(|e| format!("Deserialization failed: {}", e));
        Ok(())
    }

    #[cfg(not(feature = "serde"))]
    pub fn serialize_registry(&self) -> Result<Vec<u8>, String> {
        Err("serde feature not enabled".to_string())
    }

    #[cfg(not(feature = "serde"))]
    pub fn deserialize_registry(&mut self, _data: &[u8]) -> Result<(), String> {
        Err("serde feature not enabled".to_string())
    }
}