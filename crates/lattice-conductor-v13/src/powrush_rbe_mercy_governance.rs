// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

use crate::mercy_integration::MercyIntegration;
use crate::powrush_rbe_mercy_arbitration::RbeProposal;
use std::collections::HashMap;

// Previous structs (DynamicMercyAlignment, MercyGatedReFiProposal, etc.) assumed present

/// Enhanced Conviction Stake with exponential accumulation + mercy weighting
#[derive(Debug, Clone)]
pub struct EnhancedConvictionStake {
    pub voter_id: String,
    pub proposal_id: String,
    pub base_stake: f64,
    pub current_conviction: f64,
    pub last_update_time: u64,
    pub mercy_alignment_at_stake: f64,
}

pub fn calculate_exponential_conviction(previous: f64, new_stake: f64, alpha: f64) -> f64 {
    previous * alpha + new_stake * (1.0 - alpha)
}

pub fn get_mercy_weighted_alpha(base_alpha: f64, mercy_alignment: f64) -> f64 {
    let adjustment = (mercy_alignment - 0.7).max(0.0) * 0.08;
    (base_alpha - adjustment).clamp(0.80, 0.98)
}

/// Main function: Stake conviction with exponential growth and mercy weighting
pub fn stake_enhanced_conviction(
    integration: &mut MercyIntegration,
    proposal: &MercyGatedReFiProposal,
    voter_id: &str,
    additional_stake: f64,
    alignment: &mut DynamicMercyAlignment,
    current_time: u64,
    base_alpha: f64,
) -> Result<EnhancedConvictionStake, crate::error::MercyError> {
    let mut scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.91)).collect();

    if proposal.regenerative_impact_score > 0.85 {
        if let Some(s) = scores.get_mut(&11) { *s = 0.97; }
        if let Some(s) = scores.get_mut(&17) { *s = 0.98; }
        if let Some(s) = scores.get_mut(&22) { *s = 0.97; }
    }

    integration.evaluate_proposal(&scores)?;

    let effective_alpha = get_mercy_weighted_alpha(base_alpha, alignment.current_alignment);
    let previous_conviction = 0.0; // In production this would be loaded from state

    let new_conviction = calculate_exponential_conviction(previous_conviction, additional_stake, effective_alpha);

    alignment.restore_through_action(proposal.regenerative_impact_score, current_time);

    if new_conviction > 3.5 && proposal.regenerative_impact_score > 0.90 {
        integration.council_13_batch_tune(&[(11, 0.95), (17, 0.96)])?;
    }

    Ok(EnhancedConvictionStake {
        voter_id: voter_id.to_string(),
        proposal_id: proposal.base.id.clone(),
        base_stake: additional_stake,
        current_conviction: new_conviction,
        last_update_time: current_time,
        mercy_alignment_at_stake: alignment.current_alignment,
    })
}