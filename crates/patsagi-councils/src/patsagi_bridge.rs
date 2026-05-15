// PATSAGi Bridge — ProposalHandler trait + council specializations
// Ra-Thor monorepo (AG-SML v1.0)
// Mercy-gated, TOLC-aligned, eternally self-evolving

use crate::LatticeConductor;

pub trait ProposalHandler {
    fn handle(&mut self, proposal: &str) -> String;
}

/// Bridge between PATSAGi Councils and the Lattice Conductor.
pub struct PatsagiBridge {
    conductor: LatticeConductor,
}

impl PatsagiBridge {
    pub fn new() -> Self {
        Self {
            conductor: LatticeConductor::new(),
        }
    }

    pub fn submit_proposal(&mut self, proposal: &str) -> String {
        if !self.conductor.mercy.mercy_gate_audit(proposal) {
            return format!("PATSAGi Proposal REJECTED by Mercy Gates: {}", proposal);
        }
        let result = self.conductor.tick(proposal).unwrap_or_else(|e| e);
        format!("PATSAGi Proposal EXECUTED: {} | Result: {}", proposal, result)
    }
}

impl ProposalHandler for PatsagiBridge {
    fn handle(&mut self, proposal: &str) -> String {
        self.submit_proposal(proposal)
    }
}

// === Specific Council Types ===

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

impl ProposalHandler for EvolutionCouncil {
    fn handle(&mut self, proposal: &str) -> String {
        let enhanced = format!("SELF-EVOLUTION: {}", proposal);
        self.bridge.submit_proposal(&enhanced)
    }
}