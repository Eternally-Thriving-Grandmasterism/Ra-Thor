//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! feat/patsagi-governance-v2

use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;

pub struct AuditSigner {
    signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl AuditSigner {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }
}

#[derive(Debug, Clone)]
pub struct CouncilVoteRecord {
    pub timestamp: u64,
    pub council: String,
    pub valence_contribution: f64,
    pub approved: bool,
    pub vetoed: bool,
    pub effective_weight: f64,
    pub reputation_at_time: f64,
    pub signature: Vec<u8>,
}

impl LatticeAlchemicalEvolution {
    pub fn log_council_vote(&mut self, mut record: CouncilVoteRecord) {
        if let Some(signer) = &self.audit_signer {
            let message = self.create_audit_message(&record);
            let sig = signer.sign(&message);
            record.signature = sig.to_bytes().to_vec();
        }
        self.vote_history.push(record);
    }

    fn create_audit_message(&self, record: &CouncilVoteRecord) -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(&record.timestamp.to_le_bytes());
        msg.extend_from_slice(record.council.as_bytes());
        msg.extend_from_slice(&record.valence_contribution.to_le_bytes());
        msg.push(record.approved as u8);
        msg.push(record.vetoed as u8);
        msg.extend_from_slice(&record.effective_weight.to_le_bytes());
        msg
    }

    pub fn verify_audit_log(&self) -> bool {
        if let Some(signer) = &self.audit_signer {
            for record in &self.vote_history {
                let message = self.create_audit_message(record);
                if let Ok(sig) = Signature::from_bytes(&record.signature) {
                    if signer.verify(&message, &sig).is_err() {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }
}