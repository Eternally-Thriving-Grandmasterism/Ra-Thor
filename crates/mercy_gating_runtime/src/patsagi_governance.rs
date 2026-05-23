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

    pub fn with_policy(policy: SlashingPolicy) -> Self {
        Self {
            registry: CouncilRegistry::new(),
            policy,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CouncilArbitrationSession, PatsagiCouncil, CouncilStake};

    #[test]
    fn test_run_arbitration_applies_bonus_on_success() {
        let mut gov = PatsagiGovernance::new();
        gov.registry.upsert_stake(CouncilStake { council_id: 13, amount: 100, locked_until_turn: 0 });
        gov.registry.upsert_stake(CouncilStake { council_id: 24, amount: 100, locked_until_turn: 0 });

        let mut session = CouncilArbitrationSession::new(1000);
        session.add_council(PatsagiCouncil { id: 13, name: "A".into() });
        session.add_council(PatsagiCouncil { id: 24, name: "B".into() });
        session.propose_ma_at_increase(13, 750.0, "Test".into());
        session.propose_ma_at_increase(24, 760.0, "Support".into());

        let accepted = gov.run_arbitration(&session, 1000);

        let stake13 = gov.registry.get_stake(13).unwrap().amount;
        assert!(stake13 > 100, "Successful council should receive bonus");
    }

    #[test]
    fn test_run_arbitration_slashes_low_stake() {
        let mut gov = PatsagiGovernance::new();
        gov.registry.upsert_stake(CouncilStake { council_id: 99, amount: 5, locked_until_turn: 0 });

        let mut session = CouncilArbitrationSession::new(1001);
        session.add_council(PatsagiCouncil { id: 99, name: "LowStake".into() });
        session.propose_ma_at_increase(99, 900.0, "Risky".into());

        let _ = gov.run_arbitration(&session, 1001);

        let final_stake = gov.registry.get_stake(99).unwrap().amount;
        assert!(final_stake < 5, "Low-stake high-impact proposal should be slashed");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut gov = PatsagiGovernance::new();
        gov.registry.upsert_stake(CouncilStake { council_id: 42, amount: 777, locked_until_turn: 0 });

        #[cfg(feature = "serde")]
        {
            let bytes = gov.serialize_registry().expect("serialize should work");
            let mut gov2 = PatsagiGovernance::new();
            gov2.deserialize_registry(&bytes).expect("deserialize should work");
            assert_eq!(gov2.registry.get_stake(42).unwrap().amount, 777);
        }
    }
}