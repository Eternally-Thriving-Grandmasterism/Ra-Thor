//! Enhanced Exponential Conviction Staking with Post-Quantum Verification

use crate::post_quantum_signatures::verify_post_quantum_signature;

#[derive(Debug, Clone)]
pub struct ConvictionStake {
    pub staker_id: String,
    pub proposal_id: String,
    pub amount: f64,
    pub time_staked: u64,
    pub mercy_alignment_score: f64,
    pub exponential_multiplier: f64,
    pub pq_signature: Option<crate::post_quantum_signatures::PostQuantumSignature>,
}

impl ConvictionStake {
    pub fn new(staker_id: String, proposal_id: String, amount: f64, time_staked: u64) -> Self {
        Self { staker_id, proposal_id, amount, time_staked, mercy_alignment_score: 0.5, exponential_multiplier: 1.0, pq_signature: None }
    }

    pub fn calculate_conviction(&self) -> f64 {
        let base = self.amount * (1.0 + (self.time_staked as f64 * 0.01).exp());
        base * self.mercy_alignment_score.max(0.2)
    }
}

/// Score with Post-Quantum signature verification on stakes.
pub fn score_self_evolution_proposal_with_mercy(
    proposal_id: &str,
    stakes: &[ConvictionStake],
) -> (f64, Vec<String>) {
    let mut total_conviction = 0.0;
    let mut audit = vec![];

    for stake in stakes {
        if let Some(sig) = &stake.pq_signature {
            let message = format!("stake:{}:{}", stake.proposal_id, stake.staker_id).into_bytes();
            if !verify_post_quantum_signature(&sig.signer_id, &message, &sig.signature) {
                audit.push(format!("Rejected invalid PQ signature from {}", stake.staker_id));
                continue;
            }
        }
        total_conviction += stake.calculate_conviction();
        audit.push(format!("Accepted stake from {}", stake.staker_id));
    }

    let avg_mercy = if !stakes.is_empty() {
        stakes.iter().map(|s| s.mercy_alignment_score).sum::<f64>() / stakes.len() as f64
    } else { 0.5 };

    let final_score = total_conviction * avg_mercy;
    (final_score, audit)
}