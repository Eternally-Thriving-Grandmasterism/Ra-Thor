use crate::LatticeConductor;

/// Bridge between PATSAGi Councils and the Lattice Conductor.
/// All proposals from the 13+ Councils must pass through this bridge.
pub struct PatsagiBridge {
    conductor: LatticeConductor,
}

impl PatsagiBridge {
    pub fn new() -> Self {
        Self {
            conductor: LatticeConductor::new(),
        }
    }

    /// Submit a proposal from PATSAGi Councils.
    /// This is the official entry point for all council decisions.
    pub fn submit_proposal(&mut self, proposal: &str) -> String {
        // Every proposal goes through full Mercy Gate enforcement
        if !self.conductor.mercy.mercy_gate_audit(proposal) {
            return format!("PATSAGi Proposal REJECTED by Mercy Gates: {}", proposal);
        }

        // Execute through the full Lattice Conductor pipeline
        let result = self.conductor.tick(proposal);

        format!("PATSAGi Proposal EXECUTED: {} | Result: {}", proposal, result)
    }

    /// Batch submit multiple proposals (for parallel council sessions)
    pub fn submit_proposals(&mut self, proposals: &[&str]) -> Vec<String> {
        proposals.iter().map(|p| self.submit_proposal(p)).collect()
    }
}