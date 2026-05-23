//! PATSAGi Arbitration with reputation-based slashing as default + recovery

// ... existing code ...

/// Reputation-aware slashing is now the recommended/default path
pub fn calculate_reputation_aware_slash(
    policy: &SlashingPolicy,
    proposal: &CouncilTuningProposal,
    reputation: Option<&CouncilReputation>,
) -> u64 {
    let base = match &proposal.target {
        TuningTarget::MaAtThreshold => policy.ma_at_slash,
        TuningTarget::GateThreshold { .. } => policy.gate_slash,
        _ => policy.low_impact_slash,
    };

    if let Some(rep) = reputation {
        let multiplier = (1.0 - rep.reputation_score()).clamp(0.3, 1.0);
        return (base as f64 * multiplier) as u64;
    }
    base
}

/// Call this after proposals are accepted to recover/build reputation
pub fn record_success(reputation: &mut CouncilReputation) {
    reputation.success_count += 1;
}

/// Optional light recovery (can be called periodically or after good behavior)
pub fn apply_light_recovery(reputation: &mut CouncilReputation, amount: u64) {
    reputation.success_count = reputation.success_count.saturating_add(amount);
}