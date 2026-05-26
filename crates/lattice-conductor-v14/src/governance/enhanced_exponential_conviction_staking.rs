//! Enhanced Exponential Conviction Staking with Post-Quantum Signature Support

use crate::post_quantum_signatures::{PostQuantumSignature, create_post_quantum_signature};

#[derive(Debug, Clone)]
pub struct ConvictionStake {
    pub staker_id: String,
    pub proposal_id: String,
    pub amount: f64,
    pub time_staked: u64,
    pub mercy_alignment_score: f64,
    pub exponential_multiplier: f64,
    pub pq_signature: Option<PostQuantumSignature>,
}

impl ConvictionStake {
    pub fn new(staker_id: String, proposal_id: String, amount: f64, time_staked: u64) -> Self {
        Self {
            staker_id, proposal_id, amount, time_staked,
            mercy_alignment_score: 0.5,
            exponential_multiplier: 1.0,
            pq_signature: None,
        }
    }

    pub fn calculate_conviction(&self) -> f64 {
        let base = self.amount * (1.0 + (self.time_staked as f64 * 0.01).exp());
        base * self.mercy_alignment_score.max(0.2)
    }

    pub fn sign_with_post_quantum(&mut self) {
        let message = format!("stake:{}:{}", self.proposal_id, self.staker_id).into_bytes();
        self.pq_signature = Some(create_post_quantum_signature(&self.staker_id, &message));
    }
}