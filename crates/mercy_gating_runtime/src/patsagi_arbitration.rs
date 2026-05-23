//! PATSAGi Council Arbitration Layer with Stake requirements

use crate::{CouncilTuningProposal, TuningTarget};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct PatsagiCouncil {
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct CouncilStake {
    pub council_id: u32,
    pub amount: u64,
    pub locked_until_turn: u64,
}

impl CouncilStake {
    pub fn can_propose(&self, target: &TuningTarget) -> bool {
        match target {
            TuningTarget::MaAtThreshold => self.amount >= 50,
            TuningTarget::GateThreshold { .. } => self.amount >= 30,
            _ => self.amount >= 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CouncilArbitrationSession {
    pub turn: u64,
    pub participating_councils: Vec<PatsagiCouncil>,
    pub proposals: Vec<CouncilTuningProposal>,
    pub min_consensus: usize,
}

impl CouncilArbitrationSession {
    pub fn new(turn: u64) -> Self {
        Self {
            turn,
            participating_councils: vec![],
            proposals: vec![],
            min_consensus: 2,
        }
    }

    pub fn with_min_consensus(mut self, min: usize) -> Self {
        self.min_consensus = min;
        self
    }

    pub fn add_council(&mut self, council: PatsagiCouncil) {
        self.participating_councils.push(council);
    }

    pub fn propose(&mut self, proposal: CouncilTuningProposal) {
        self.proposals.push(proposal);
    }

    pub fn has_consensus(&self) -> bool {
        let unique: HashSet<u32> = self.proposals.iter().map(|p| p.council_id).collect();
        unique.len() >= self.min_consensus
    }

    /// Returns proposals that both reached consensus AND have sufficient stake
    pub fn accepted_proposals_with_staking(
        &self,
        stakes: &HashMap<u32, CouncilStake>,
    ) -> Vec<CouncilTuningProposal> {
        if !self.has_consensus() {
            return vec![];
        }

        self.proposals
            .iter()
            .filter(|p| {
                stakes.get(&p.council_id)
                    .map_or(false, |stake| stake.can_propose(&p.target))
            })
            .cloned()
            .collect()
    }

    pub fn propose_ma_at_increase(&mut self, council_id: u32, new_value: f64, justification: String) {
        self.propose(CouncilTuningProposal {
            council_id,
            target: TuningTarget::MaAtThreshold,
            new_value,
            justification,
            proposed_at_turn: self.turn,
        });
    }

    pub fn propose_gate_tightening(&mut self, council_id: u32, gate: String, new_value: f64, justification: String) {
        self.propose(CouncilTuningProposal {
            council_id,
            target: TuningTarget::GateThreshold { gate },
            new_value,
            justification,
            proposed_at_turn: self.turn,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staking_filters_low_stake_proposals() {
        let mut session = CouncilArbitrationSession::new(200);
        session.add_council(PatsagiCouncil { id: 13, name: "Coherence".into() });
        session.add_council(PatsagiCouncil { id: 24, name: "Unity".into() });

        session.propose_ma_at_increase(13, 755.0, "High impact".into());

        let mut stakes = HashMap::new();
        stakes.insert(13, CouncilStake { council_id: 13, amount: 20, locked_until_turn: 0 });
        stakes.insert(24, CouncilStake { council_id: 24, amount: 60, locked_until_turn: 0 });

        let accepted = session.accepted_proposals_with_staking(&stakes);
        assert!(accepted.is_empty()); // Council 13 stake too low for MaAtThreshold
    }
}