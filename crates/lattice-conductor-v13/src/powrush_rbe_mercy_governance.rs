// Enhanced Conviction Staking with Exponential Accumulation + Mercy-Weighted Growth
// Thunder Lattice Governance v2 (enhanced)

use crate::mercy_integration::MercyIntegration;
use crate::powrush_rbe_mercy_arbitration::RbeProposal;
use std::collections::HashMap;

// ... (keeping previous structs for brevity in this thinking step)

/// Enhanced Conviction Stake with exponential accumulation
#[derive(Debug, Clone)]
pub struct EnhancedConvictionStake {
    pub voter_id: String,
    pub proposal_id: String,
    pub base_stake: f64,
    pub current_conviction: f64,
    pub last_update_time: u64,
    pub mercy_alignment_at_stake: f64,
}

/// Calculate new conviction using exponential smoothing (Commons Stack style)
/// alpha closer to 1.0 = slower, more patient conviction growth
pub fn calculate_exponential_conviction(
    previous_conviction: f64,
    new_stake: f64,
    alpha: f64,
) -> f64 {
    previous_conviction * alpha + new_stake * (1.0 - alpha)
}

/// Mercy-weighted conviction growth rate
/// Higher mercy alignment = faster conviction accumulation (gentle reward)
pub fn get_mercy_weighted_alpha(base_alpha: f64, mercy_alignment: f64) -> f64 {
    // Higher alignment -> slightly lower alpha (faster growth)
    let adjustment = (mercy_alignment - 0.7).max(0.0) * 0.08;
    (base_alpha - adjustment).clamp(0.80, 0.98)
}

// Additional functions for staking with living mercy alignment would go here...
// (Full enhanced implementation would replace/extend previous ConvictionStake logic)