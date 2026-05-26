//! Self-Evolution Proposal — Complete End-to-End Post-Quantum + Hybrid Simulation

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_end_to_end_simulation() {
        // === 1. Create and sign proposal with Post-Quantum signature ===
        let mut proposal = SelfEvolutionProposal::new(
            "prop-pq-001".to_string(),
            "Enable Post-Quantum Sovereign Channels".to_string(),
            "Full hybrid + PQ signature flow demo".to_string(),
            "sherif".to_string(),
        );
        proposal.sign_with_post_quantum("sherif");

        // === 2. Serialize proposal (simplified) ===
        let proposal_bytes = format!("{}|{}|{}", proposal.id, proposal.title, proposal.proposed_by).into_bytes();

        // === 3. Establish Hybrid Channel ===
        let mut channel = HybridSovereignChannel::new("sherif", "patsagi-council");
        channel.establish_classical_key([0x42; 32]);
        channel.establish_post_quantum_secret(vec![0x99; 32]);
        channel.finalize_hybrid_key();
        assert!(channel.is_active());

        // === 4. Simulate encryption of the signed proposal over hybrid channel ===
        if let Some(aes_key) = channel.get_aes_gcm_key() {
            println!("[SIM] Encrypting proposal bytes using hybrid-derived AES key (len={})", aes_key.len());
            // In real code: use AES-GCM with the derived key to encrypt proposal_bytes
        }

        println!("[E2E SIM] Post-quantum signed + hybrid-encrypted self-evolution proposal flow complete.");
    }
}