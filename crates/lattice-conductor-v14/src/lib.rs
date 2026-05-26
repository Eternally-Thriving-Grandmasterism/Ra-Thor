//! Lattice Conductor v14
//! Exposes governance, hybrid crypto, post-quantum features, and self-evolution.

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod governance;
pub mod hybrid_sovereign_channel;
pub mod post_quantum_signatures;
pub mod crypto_traits;
pub mod self_evolution;   // NEW

pub use governance::self_evolution_proposal::SelfEvolutionProposal;
pub use post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};
pub use hybrid_sovereign_channel::HybridSovereignChannel;
pub use self_evolution::{SelfEvolutionLoop, submit_self_evolution_proposal_securely};

use std::sync::atomic::{AtomicBool, Ordering};

pub struct LatticeConductorV14 {
    pub cosmic_loop_ready: AtomicBool,
}

impl LatticeConductorV14 {
    pub fn new() -> Self {
        Self { cosmic_loop_ready: AtomicBool::new(true) }
    }

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
}