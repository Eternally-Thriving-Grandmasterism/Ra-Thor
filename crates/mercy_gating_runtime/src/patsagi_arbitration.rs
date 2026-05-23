//! PATSAGi Council Arbitration Layer (enhanced with simple consensus)
//!
//! Councils must reach a minimum number of distinct participants before
//! their proposals are accepted and forwarded to MercyGatingRuntime.

use crate::{CouncilTuningProposal, TuningTarget};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct PatsagiCouncil {
    pub id: u32,
    pub name: String,
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

    /// Simple consensus: at least `min_consensus` distinct councils must have proposed.
    pub fn has_consensus(&self) -> bool {
        let unique: HashSet<u32> = self.proposals.iter().map(|p| p.council_id).collect();
        unique.len() >= self.min_consensus
    }

    pub fn accepted_proposals(&self) -> Vec<CouncilTuningProposal> {
        if self.has_consensus() {
            self.proposals.clone()
        } else {
            vec![]
        }
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
    fn test_consensus_requires_multiple_councils() {
        let mut session = CouncilArbitrationSession::new(100).with_min_consensus(2);
        session.add_council(PatsagiCouncil { id: 13, name: "Coherence".into() });
        session.propose_ma_at_increase(13, 750.0, "Test".into());
        assert!(!session.has_consensus());

        session.add_council(PatsagiCouncil { id: 24, name: "Unity".into() });
        session.propose_ma_at_increase(24, 760.0, "Support".into());
        assert!(session.has_consensus());
        assert_eq!(session.accepted_proposals().len(), 2);
    }
}