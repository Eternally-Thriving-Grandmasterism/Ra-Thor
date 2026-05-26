//! Lattice Conductor v14 — Secure Governance Submission Workflow

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod governance;
pub mod hybrid_sovereign_channel;
pub mod post_quantum_signatures;

pub use governance::self_evolution_proposal::SelfEvolutionProposal;
pub use post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};
pub use hybrid_sovereign_channel::HybridSovereignChannel;

use std::sync::atomic::{AtomicBool, Ordering};

pub struct LatticeConductorV14 {
    pub cosmic_loop_ready: AtomicBool,
}

impl LatticeConductorV14 {
    pub fn new() -> Self {
        Self { cosmic_loop_ready: AtomicBool::new(true) }
    }

    /// High-level secure governance submission workflow.
    /// Verifies PQ signature, optional stake/vote signatures, and runs full evaluation.
    pub fn submit_secure_governance_proposal(
        &self,
        proposal: &SelfEvolutionProposal,
        threshold: f64,
    ) -> (bool, Vec<String>, f64) {
        println!("[LATTICE CONDUCTOR] Secure governance submission started for {}", proposal.id);

        // This already includes PQ signature verification inside evaluate_governance
        let result = proposal.evaluate_governance(threshold);

        println!("[LATTICE CONDUCTOR] Secure submission complete.");
        result
    }

    pub fn submit_signed_self_evolution_proposal(&self, proposal: &SelfEvolutionProposal) -> (bool, Vec<String>, f64) {
        proposal.evaluate_governance(5.0)
    }

    pub fn create_hybrid_sovereign_channel(&self, from: &str, to: &str) -> HybridSovereignChannel {
        HybridSovereignChannel::new(from, to)
    }

    pub fn verify_pq_signature(&self, signer_id: &str, message_hash: &[u8], signature: &[u8]) -> bool {
        verify_post_quantum_signature(signer_id, message_hash, signature)
    }
}