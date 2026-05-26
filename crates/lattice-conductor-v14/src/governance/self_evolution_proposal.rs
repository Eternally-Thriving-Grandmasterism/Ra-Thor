//! Self-Evolution Proposal — Complete Simulation with Actual AES-GCM

use crate::post_quantum_signatures::create_post_quantum_signature;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_simulation_with_actual_aes_gcm() {
        let mut proposal = SelfEvolutionProposal::new(
            "prop-aes-01".to_string(),
            "Full AES-GCM + PQ Demo".to_string(),
            "...".to_string(),
            "sherif".to_string(),
        );
        proposal.sign_with_post_quantum("sherif");

        let proposal_bytes = format!("{}|{}", proposal.id, proposal.title).into_bytes();

        let mut channel = HybridSovereignChannel::new("sherif", "council");
        channel.establish_classical_key([0x42; 32]);
        channel.establish_post_quantum_secret(vec![0x99; 32]);
        channel.finalize_hybrid_key();

        if let Some(key_bytes) = channel.get_aes_gcm_key() {
            let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key_bytes));
            let nonce = Nonce::<Aes256Gcm>::from_slice(&[0u8; 12]);

            match cipher.encrypt(nonce, Payload { msg: &proposal_bytes, aad: &[] }) {
                Ok(ciphertext) => {
                    println!("[SIM] Encrypted {} bytes proposal -> {} bytes ciphertext", proposal_bytes.len(), ciphertext.len());
                }
                Err(_) => panic!("Encryption failed"),
            }
        }
    }
}