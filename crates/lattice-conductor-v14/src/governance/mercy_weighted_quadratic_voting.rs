//! Mercy-Weighted Quadratic Voting with Post-Quantum Signature Verification

use crate::post_quantum_signatures::verify_post_quantum_signature;

#[derive(Debug, Clone)]
pub struct MercyWeightedVote {
    pub voter_id: String,
    pub proposal_id: String,
    pub raw_power: f64,
    pub mercy_alignment: f64,
    pub conviction_multiplier: f64,
    pub pq_signature: Option<crate::post_quantum_signatures::PostQuantumSignature>,
}

impl MercyWeightedVote {
    pub fn new(voter_id: String, proposal_id: String, raw_power: f64) -> Self {
        Self { voter_id, proposal_id, raw_power, mercy_alignment: 0.5, conviction_multiplier: 1.0, pq_signature: None }
    }
}

/// Tally with optional Post-Quantum signature verification.
pub fn tally_mercy_weighted_quadratic_votes(votes: &[MercyWeightedVote]) -> (f64, Vec<String>) {
    let mut total = 0.0;
    let mut audit = vec![];

    for vote in votes {
        if let Some(sig) = &vote.pq_signature {
            let message = format!("vote:{}:{}", vote.proposal_id, vote.voter_id).into_bytes();
            if !verify_post_quantum_signature(&sig.signer_id, &message, &sig.signature) {
                audit.push(format!("Rejected invalid PQ signature from {}", vote.voter_id));
                continue;
            }
        }
        total += vote.raw_power;
        audit.push(format!("Accepted vote from {}", vote.voter_id));
    }

    (total, audit)
}