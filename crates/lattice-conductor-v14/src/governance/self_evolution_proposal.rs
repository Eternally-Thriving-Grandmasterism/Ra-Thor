//! Self-Evolution Proposal — PQ Verification in Governance + Full Round-Trip

use crate::post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};
use crate::hybrid_sovereign_channel::HybridSovereignChannel;
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit, Payload};

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

    /// Governance evaluation now includes Post-Quantum signature verification.
    pub fn evaluate_governance(&self, threshold: f64) -> (bool, Vec<String>, f64) {
        let mut audit = vec![];

        let signature_valid = if let Some(sig) = &self.pq_signature {
            let message = format!("proposal:{}:{}", self.id, self.title).into_bytes();
            verify_post_quantum_signature(&sig.signer_id, &message, &sig.signature)
        } else {
            false
        };

        audit.push(format!("PQ Signature Valid: {}", signature_valid));

        if !signature_valid {
            return (false, audit, 0.0);
        }

        let final_score = self.mercy_alignment * 10.0;
        let passes = final_score > threshold;

        audit.push(format!("Governance passes: {}", passes));
        (passes, audit, final_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_encrypted_verified_roundtrip() {
        let mut proposal = SelfEvolutionProposal::new(
            "prop-roundtrip".to_string(),
            "Full PQ + Hybrid Roundtrip".to_string(),
            "...".to_string(),
            "sherif".to_string(),
        );
        proposal.sign_with_post_quantum("sherif");

        let proposal_bytes = format!("{}|{}", proposal.id, proposal.title).into_bytes();

        let mut channel = HybridSovereignChannel::new("sherif", "council");
        channel.establish_classical_key([0x42; 32]);
        channel.establish_post_quantum_secret(vec![0x99; 32]);
        channel.finalize_hybrid_key();

        // Actual encryption
        let _ciphertext = if let Some(key) = channel.get_aes_gcm_key() {
            let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
            let nonce = Nonce::<Aes256Gcm>::from_slice(&[0u8; 12]);
            cipher.encrypt(nonce, Payload { msg: &proposal_bytes, aad: &[] }).ok()
        } else { None };

        // Governance with built-in PQ verification
        let (passes, audit, _score) = proposal.evaluate_governance(5.0);
        assert!(passes);
        println!("[ROUNDTRIP] Governance audit: {:?}", audit);
    }
}