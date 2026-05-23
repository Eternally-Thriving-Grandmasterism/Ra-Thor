//! PATSAGi Council Arbitration Layer (initial sketch)
//!
//! This module models how multiple PATSAGi Councils can deliberate and
//! produce `CouncilTuningProposal`s that are then applied to `MercyGatingRuntime`.
//! Future: full multi-council consensus, justification validation, and
//! hot-reload triggering via Lattice Conductor.

use crate::{CouncilTuningProposal, TuningTarget};

/// Represents a single PATSAGi Council participating in arbitration.
#[derive(Debug, Clone)]
pub struct PatsagiCouncil {
    pub id: u32,
    pub name: String,
}

/// A session where councils deliberate and produce tuning proposals.
#[derive(Debug, Clone)]
pub struct CouncilArbitrationSession {
    pub turn: u64,
    pub participating_councils: Vec<PatsagiCouncil>,
    pub proposals: Vec<CouncilTuningProposal>,
}

impl CouncilArbitrationSession {
    pub fn new(turn: u64) -> Self {
        Self {
            turn,
            participating_councils: vec![],
            proposals: vec![],
        }
    }

    pub fn add_council(&mut self, council: PatsagiCouncil) {
        self.participating_councils.push(council);
    }

    /// Example: Council #13 proposes raising Ma'at threshold during high-coherence arbitration.
    pub fn propose_ma_at_increase(&mut self, council_id: u32, new_value: f64, justification: String) {
        self.proposals.push(CouncilTuningProposal {
            council_id,
            target: TuningTarget::MaAtThreshold,
            new_value,
            justification,
            proposed_at_turn: self.turn,
        });
    }

    /// Example: A council proposes tightening a specific high-impact gate.
    pub fn propose_gate_tightening(&mut self, council_id: u32, gate: String, new_value: f64, justification: String) {
        self.proposals.push(CouncilTuningProposal {
            council_id,
            target: TuningTarget::GateThreshold { gate },
            new_value,
            justification,
            proposed_at_turn: self.turn,
        });
    }

    /// Returns the collected proposals ready to be applied to MercyGatingRuntime.
    pub fn collect_proposals(&self) -> &[CouncilTuningProposal] {
        &self.proposals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arbitration_session_collects_proposals() {
        let mut session = CouncilArbitrationSession::new(42);
        session.add_council(PatsagiCouncil { id: 13, name: "Council of Coherence".to_string() });
        session.propose_ma_at_increase(13, 755.0, "Raised during sacred node arbitration".to_string());

        assert_eq!(session.collect_proposals().len(), 1);
    }
}