//! Self-Evolution Proposal with Post-Quantum Signature Support + End-to-End Example

use crate::post_quantum_signatures::create_post_quantum_signature;
use crate::hybrid_sovereign_channel::HybridSovereignChannel;

#[derive(Debug, Clone)]
pub struct SelfEvolutionProposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub proposed_by: String,
    pub mercy_alignment: f64,
    pub pq_signature: Option<crate::post_quantum_signatures::PostQuantumSignature>,
}

impl SelfEvolutionProposal {
    pub fn new(id: String, title: String, description: String, proposed_by: String) -> Self {
        Self { id, title, description, proposed_by, mercy_alignment: 0.5, pq_signature: None }
    }

    pub fn sign_with_post_quantum(&mut self, signer_id: &str) {
        let message = format!("proposal:{}:{}", self.id, self.title).into_bytes();
        self.pq_signature = Some(create_post_quantum_signature(signer_id, &message));
    }
}

// ==================== FULL END-TO-END EXAMPLE ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_end_to_end_pq_signed_hybrid_proposal() {
        // 1. Create Self-Evolution Proposal
        let mut proposal = SelfEvolutionProposal::new(
            "prop-042".to_string(),
            "Integrate Post-Quantum Sovereign Channels".to_string(),
            "Enable hybrid AES-GCM + Kyber channels across the lattice.".to_string(),
            "sherif".to_string(),
        );

        // 2. Sign with Post-Quantum Signature
        proposal.sign_with_post_quantum("sherif");
        assert!(proposal.pq_signature.is_some());

        // 3. Establish Hybrid Channel (Classical + Post-Quantum)
        let mut channel = HybridSovereignChannel::new("sherif", "patsagi-council");
        channel.establish_classical_key([0xAA; 32]);
        channel.establish_post_quantum_secret(vec![0xBB; 32]);
        channel.finalize_hybrid_key();
        assert!(channel.is_active());

        println!("[E2E] Post-quantum signed proposal ready for hybrid encrypted transmission.");
    }
}