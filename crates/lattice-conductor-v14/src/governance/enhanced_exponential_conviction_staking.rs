//! Enhanced Exponential Conviction Staking — Phase 14.1 Governance Primitive
//! Dedicated module with mercy-alignment metadata and auditable scoring.

/// Conviction stake with mercy alignment metadata.
#[derive(Debug, Clone)]
pub struct ConvictionStake {
    pub staker_id: String,
    pub proposal_id: String,
    pub amount: f64,
    pub time_staked: u64,           // In blocks or seconds
    pub mercy_alignment_score: f64, // From TOLC 7 Gates + council review
    pub exponential_multiplier: f64,
}

impl ConvictionStake {
    pub fn calculate_conviction(&self) -> f64 {
        // Enhanced exponential: conviction grows exponentially with time, modulated by mercy
        let base = self.amount * (1.0 + (self.time_staked as f64 * 0.01).exp());
        base * self.mercy_alignment_score.max(0.2) // Mercy floor
    }

    pub fn to_mercy_metadata(&self) -> String {
        format!(
            "stake:{}|amount:{:.2}|time:{}|mercy:{:.3}|conviction:{:.2}",
            self.staker_id, self.amount, self.time_staked, self.mercy_alignment_score, self.calculate_conviction()
        )
    }
}

/// Applies exponential conviction with mercy weighting.
pub fn apply_enhanced_exponential_conviction(stakes: &[ConvictionStake]) -> Vec<(String, f64)> {
    stakes
        .iter()
        .map(|s| (s.proposal_id.clone(), s.calculate_conviction()))
        .collect()
}

/// Auditable scoring for self-evolution loop proposals.
pub fn score_self_evolution_proposal_with_mercy(
    proposal_id: &str,
    stakes: &[ConvictionStake],
) -> (f64, Vec<String>) {
    let convictions: Vec<f64> = stakes.iter().map(|s| s.calculate_conviction()).collect();
    let total_conviction: f64 = convictions.iter().sum();
    let avg_mercy: f64 = if !stakes.is_empty() {
        stakes.iter().map(|s| s.mercy_alignment_score).sum::<f64>() / stakes.len() as f64
    } else {
        0.5
    };

    let final_score = total_conviction * avg_mercy;
    let metadata: Vec<String> = stakes.iter().map(|s| s.to_mercy_metadata()).collect();

    (final_score, metadata)
}