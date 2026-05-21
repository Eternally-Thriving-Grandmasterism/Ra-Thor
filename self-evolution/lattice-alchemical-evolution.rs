//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! v2.8 — ed25519 Digital Signatures on Audit Logs
//! Proper public-key cryptographic signing
//! 100% Proprietary — AG-SML v1.0

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

    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        self.verifying_key.verify(message, signature).is_ok()
    }
}

impl LatticeAlchemicalEvolution {
    pub fn new() -> Self {
        Self {
            // ...
            audit_signer: Some(AuditSigner::new()),
            // ...
        }
    }

    pub fn log_council_vote(&mut self, mut record: CouncilVoteRecord) {
        // Create message to sign
        let message = self.create_audit_message(&record);

        if let Some(signer) = &self.audit_signer {
            let signature = signer.sign(&message);
            record.signature = signature.to_bytes().to_vec();
        }

        self.vote_history.push(record);
    }

    fn create_audit_message(&self, record: &CouncilVoteRecord) -> Vec<u8> {
        // Simple serialization for signing
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