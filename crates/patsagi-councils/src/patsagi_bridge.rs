//! PATSAGi Bridge — v14.15.0
//!
//! ProposalHandler trait + specialized council bridges.
//! Mercy-gated, TOLC-aligned, Living Cosmic Tick ready.
//!
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::LatticeConductor;

/// Trait for any surface that can handle a governance proposal.
pub trait ProposalHandler {
    fn handle(&mut self, proposal: &str) -> String;
}

/// Bridge between PATSAGi Councils and the Lattice Conductor.
pub struct PatsagiBridge {
    conductor: LatticeConductor,
    proposals_handled: u64,
}

impl PatsagiBridge {
    pub fn new() -> Self {
        Self {
            conductor: LatticeConductor::new(),
            proposals_handled: 0,
        }
    }

    pub fn submit_proposal(&mut self, proposal: &str) -> String {
        self.proposals_handled = self.proposals_handled.saturating_add(1);

        if !self.conductor.mercy.mercy_gate_audit(proposal) {
            return format!(
                "PATSAGi Proposal REJECTED by Mercy Gates (v14.15.0): {}",
                proposal
            );
        }

        let result = self
            .conductor
            .tick(proposal)
            .unwrap_or_else(|e| e);

        format!(
            "PATSAGi Proposal EXECUTED (v14.15.0): {} | Result: {} | Living Cosmic Tick active",
            proposal, result
        )
    }

    pub fn summary(&self) -> String {
        format!(
            "PatsagiBridge v14.15.0 | proposals_handled={} | Living Cosmic Tick active",
            self.proposals_handled
        )
    }
}

impl Default for PatsagiBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl ProposalHandler for PatsagiBridge {
    fn handle(&mut self, proposal: &str) -> String {
        self.submit_proposal(proposal)
    }
}

// =============================================================================
// Specialized Council Types
// =============================================================================

pub struct ArchitecturalCouncil {
    bridge: PatsagiBridge,
}

impl ArchitecturalCouncil {
    pub fn new() -> Self {
        Self {
            bridge: PatsagiBridge::new(),
        }
    }
}

impl Default for ArchitecturalCouncil {
    fn default() -> Self {
        Self::new()
    }
}

impl ProposalHandler for ArchitecturalCouncil {
    fn handle(&mut self, proposal: &str) -> String {
        let enhanced = format!("ARCHITECTURAL: {}", proposal);
        self.bridge.submit_proposal(&enhanced)
    }
}

pub struct MercyCouncil {
    bridge: PatsagiBridge,
}

impl MercyCouncil {
    pub fn new() -> Self {
        Self {
            bridge: PatsagiBridge::new(),
        }
    }
}

impl Default for MercyCouncil {
    fn default() -> Self {
        Self::new()
    }
}

impl ProposalHandler for MercyCouncil {
    fn handle(&mut self, proposal: &str) -> String {
        let enhanced = format!("MERCY-FOCUSED: {}", proposal);
        self.bridge.submit_proposal(&enhanced)
    }
}

pub struct EvolutionCouncil {
    bridge: PatsagiBridge,
}

impl EvolutionCouncil {
    pub fn new() -> Self {
        Self {
            bridge: PatsagiBridge::new(),
        }
    }
}

impl Default for EvolutionCouncil {
    fn default() -> Self {
        Self::new()
    }
}

impl ProposalHandler for EvolutionCouncil {
    fn handle(&mut self, proposal: &str) -> String {
        let enhanced = format!("SELF-EVOLUTION: {}", proposal);
        self.bridge.submit_proposal(&enhanced)
    }
}
