//! Self-Evolution Proposal — PQ Verification in Governance (v14.8.2)

use crate::post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};

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
        Self {
            id,
            title,
            description,
            proposed_by,
            mercy_alignment: 0.5,
            pq_signature: None,
        }
    }

    pub fn sign_with_post_quantum(&mut self, signer_id: &str) {
        let message = format!("proposal:{}:{}", self.id, self.title).into_bytes();
        self.pq_signature = Some(create_post_quantum_signature(signer_id, &message));
    }

    /// Governance evaluation includes Post-Quantum signature verification.
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
    fn test_pq_signed_proposal_passes_governance() {
        let mut proposal = SelfEvolutionProposal::new(
            "prop-1".into(),
            "Cosmic Loop Hardening".into(),
            "Strengthen shared flag".into(),
            "sherif".into(),
        );
        proposal.mercy_alignment = 0.95;
        proposal.sign_with_post_quantum("sherif");

        let (passes, audit, score) = proposal.evaluate_governance(5.0);
        assert!(passes, "audit={:?} score={}", audit, score);
        assert!(score > 5.0);
    }
}
