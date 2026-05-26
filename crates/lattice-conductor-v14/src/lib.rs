//! Lattice Conductor v14
//! Provides clean, modular access to governance, hybrid channels, and post-quantum features.

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod governance;
pub mod hybrid_sovereign_channel;
pub mod post_quantum_signatures;

// Convenient re-exports
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

    /// High-level secure governance entry point.
    pub fn submit_secure_governance_proposal(
        &self,
        proposal: &SelfEvolutionProposal,
        threshold: f64,
    ) -> (bool, Vec<String>, f64) {
        proposal.evaluate_governance(threshold)
    }

    pub fn create_hybrid_sovereign_channel(&self, from: &str, to: &str) -> HybridSovereignChannel {
        HybridSovereignChannel::new(from, to)
    }

    pub fn verify_pq_signature(&self, signer_id: &str, message_hash: &[u8], signature: &[u8]) -> bool {
        verify_post_quantum_signature(signer_id, message_hash, signature)
    }
}