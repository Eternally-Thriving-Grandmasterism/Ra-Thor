// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0
// Thunder Lattice Governance — Mercy-Gated ReFi + Quadratic Voting + Conviction + Mycelial Pruning

use crate::mercy_integration::MercyIntegration;
use crate::powrush_rbe_mercy_arbitration::RbeProposal;
use std::collections::HashMap;

// === Core Governance Types ===

#[derive(Debug, Clone)]
pub struct MercyGatedReFiProposal {
    pub base: RbeProposal,
    pub regenerative_impact_score: f64,
    pub conviction_stake: f64,
    pub governance_type: ReFiGovernanceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReFiGovernanceType {
    ResourceAllocation,
    FactionAbundanceFlow,
    EcologicalStewardship,
    MercyWeightedVote,
}

// === Dynamic Mercy Alignment (Living) ===

#[derive(Debug, Clone)]
pub struct DynamicMercyAlignment {
    pub current_alignment: f64,
    pub last_active_timestamp: u64,
    pub total_merciful_actions: u64,
}

impl DynamicMercyAlignment {
    pub fn new(initial: f64) -> Self {
        Self {
            current_alignment: initial.clamp(0.3, 1.0),
            last_active_timestamp: 0,
            total_merciful_actions: 0,
        }
    }

    pub fn apply_decay(&mut self, current_time: u64, decay_rate: f64, min_floor: f64) {
        let time_since = current_time.saturating_sub(self.last_active_timestamp) as f64;
        let history_protection = (self.total_merciful_actions as f64).sqrt() * 0.0004;
        let effective_decay = (decay_rate - history_protection).max(0.00001);
        let decay_amount = time_since * effective_decay;
        self.current_alignment = (self.current_alignment - decay_amount).max(min_floor);
    }

    pub fn restore_through_action(&mut self, action_impact: f64, current_time: u64) {
        let growth = action_impact * 0.09;
        self.current_alignment = (self.current_alignment + growth).min(1.0);
        self.last_active_timestamp = current_time;
        self.total_merciful_actions += 1;
    }

    pub fn get_living_influence_multiplier(&self) -> f64 {
        0.6 + (self.current_alignment - 0.3).max(0.0) * 1.1
    }
}

// === Quadratic Voting ===

#[derive(Debug, Clone)]
pub struct MercyWeightedQuadraticVote {
    pub voter_id: String,
    pub proposal_id: String,
    pub votes_cast: u32,
    pub cost_paid: f64,
    pub mercy_alignment: f64,
    pub effective_influence: f64,
}

pub fn calculate_quadratic_vote_cost(votes: u32, base_cost: f64) -> f64 {
    (votes as f64).powi(2) * base_cost
}

pub fn cast_mercy_weighted_quadratic_vote(
    integration: &mut MercyIntegration,
    proposal: &MercyGatedReFiProposal,
    voter_id: &str,
    desired_votes: u32,
    alignment: &mut DynamicMercyAlignment,
    base_vote_cost: f64,
    current_time: u64,
) -> Result<MercyWeightedQuadraticVote, crate::error::MercyError> {
    let mut scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.90)).collect();

    if proposal.regenerative_impact_score > 0.85 {
        if let Some(s) = scores.get_mut(&11) { *s = 0.97; }
        if let Some(s) = scores.get_mut(&12) { *s = 0.96; }
        if let Some(s) = scores.get_mut(&17) { *s = 0.98; }
        if let Some(s) = scores.get_mut(&22) { *s = 0.97; }
    }

    integration.evaluate_proposal(&scores)?;

    let raw_cost = calculate_quadratic_vote_cost(desired_votes, base_vote_cost);
    let mercy_multiplier = alignment.get_living_influence_multiplier();
    let effective_influence = (desired_votes as f64).sqrt() * mercy_multiplier;

    alignment.restore_through_action(proposal.regenerative_impact_score, current_time);

    if desired_votes >= 5 && proposal.regenerative_impact_score > 0.90 {
        integration.council_13_batch_tune(&[(11, 0.95), (17, 0.96)])?;
    }

    Ok(MercyWeightedQuadraticVote {
        voter_id: voter_id.to_string(),
        proposal_id: proposal.base.id.clone(),
        votes_cast: desired_votes,
        cost_paid: raw_cost,
        mercy_alignment: alignment.current_alignment,
        effective_influence,
    })
}

// === Conviction Staking ===

#[derive(Debug, Clone)]
pub struct ConvictionStake {
    pub voter_id: String,
    pub proposal_id: String,
    pub quadratic_votes: u32,
    pub base_conviction: f64,
    pub conviction_multiplier: f64,
    pub mercy_alignment: f64,
    pub total_effective_influence: f64,
    pub staked_at: u64,
}

pub fn stake_conviction_on_quadratic_vote(
    integration: &mut MercyIntegration,
    proposal: &MercyGatedReFiProposal,
    voter_id: &str,
    quadratic_votes: u32,
    alignment: &mut DynamicMercyAlignment,
    base_vote_cost: f64,
    current_timestamp: u64,
) -> Result<ConvictionStake, crate::error::MercyError> {
    let mut scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.91)).collect();

    if proposal.regenerative_impact_score > 0.85 {
        if let Some(s) = scores.get_mut(&11) { *s = 0.97; }
        if let Some(s) = scores.get_mut(&12) { *s = 0.96; }
        if let Some(s) = scores.get_mut(&17) { *s = 0.98; }
        if let Some(s) = scores.get_mut(&22) { *s = 0.97; }
    }

    integration.evaluate_proposal(&scores)?;

    let base_conviction = (quadratic_votes as f64).sqrt();
    let time_factor = 1.0 + ((current_timestamp % 1000) as f64 / 1000.0) * 0.4;
    let mercy_boost = 1.0 + (alignment.current_alignment - 0.82).max(0.0) * 0.7;
    let conviction_multiplier = time_factor * mercy_boost;
    let total_effective_influence = base_conviction * conviction_multiplier;

    if total_effective_influence > 4.5 && proposal.regenerative_impact_score > 0.90 {
        integration.council_13_batch_tune(&[(11, 0.95), (17, 0.96), (22, 0.95)])?;
    }

    Ok(ConvictionStake {
        voter_id: voter_id.to_string(),
        proposal_id: proposal.base.id.clone(),
        quadratic_votes,
        base_conviction,
        conviction_multiplier,
        mercy_alignment: alignment.current_alignment,
        total_effective_influence,
        staked_at: current_timestamp,
    })
}

// === Mercy Recalibration (Slashing) ===

#[derive(Debug, Clone)]
pub struct MercyRecalibration {
    pub voter_id: String,
    pub proposal_id: String,
    pub original_influence: f64,
    pub recalibrated_influence: f64,
    pub mercy_alignment_before: f64,
    pub mercy_alignment_after: f64,
    pub reason: String,
    pub recalibrated_by: String,
    pub severity: f64,
}

pub fn apply_mercy_recalibration(
    integration: &mut MercyIntegration,
    vote: &mut MercyWeightedQuadraticVote,
    reason: &str,
    severity: f64,
    recalibrated_by: &str,
) -> Result<MercyRecalibration, crate::error::MercyError> {
    let mut scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.88)).collect();
    if let Some(s) = scores.get_mut(&2) { *s = 0.95; }
    if let Some(s) = scores.get_mut(&5) { *s = 0.94; }
    if let Some(s) = scores.get_mut(&6) { *s = 0.93; }

    integration.evaluate_proposal(&scores)?;

    let original = vote.effective_influence;
    let before = vote.mercy_alignment;

    let reduction = 1.0 - (severity * 0.6);
    let new_influence = (original * reduction).max(0.1);
    let new_alignment = (before * (1.0 - severity * 0.5)).max(0.3);

    vote.effective_influence = new_influence;
    vote.mercy_alignment = new_alignment;

    if severity > 0.6 {
        integration.council_13_batch_tune(&[(2, 0.96), (5, 0.95)])?;
    }

    Ok(MercyRecalibration {
        voter_id: vote.voter_id.clone(),
        proposal_id: vote.proposal_id.clone(),
        original_influence: original,
        recalibrated_influence: new_influence,
        mercy_alignment_before: before,
        mercy_alignment_after: new_alignment,
        reason: reason.to_string(),
        recalibrated_by: recalibrated_by.to_string(),
        severity,
    })
}

// === Mycelial Pruning ===

#[derive(Debug, Clone)]
pub struct MycelialPruneEvent {
    pub voter_id: String,
    pub influence_before: f64,
    pub influence_after: f64,
    pub alignment_before: f64,
    pub alignment_after: f64,
    pub reason: String,
    pub pruned_by: String,
    pub severity: f64,
}

pub fn apply_mycelial_pruning(
    integration: &mut MercyIntegration,
    alignment: &mut DynamicMercyAlignment,
    voter_id: &str,
    current_influence: f64,
    time_inactive: u64,
    past_underperformance: f64,
    pruned_by: &str,
) -> Result<MycelialPruneEvent, crate::error::MercyError> {
    let mut scores: HashMap<u8, f64> = (1..=24).map(|i| (i, 0.87)).collect();
    if let Some(s) = scores.get_mut(&2) { *s = 0.94; }
    if let Some(s) = scores.get_mut(&4) { *s = 0.93; }
    if let Some(s) = scores.get_mut(&5) { *s = 0.94; }
    if let Some(s) = scores.get_mut(&7) { *s = 0.92; }

    integration.evaluate_proposal(&scores)?;

    let before_align = alignment.current_alignment;
    let inactivity_f = (time_inactive as f64 / 1000.0).min(1.0);
    let under_f = past_underperformance;
    let low_align_f = (0.6 - before_align).max(0.0);

    let severity = ((inactivity_f * 0.4) + (under_f * 0.4) + (low_align_f * 0.2)).clamp(0.1, 0.85);

    let new_align = (before_align * (1.0 - severity * 0.5)).max(0.25);
    let new_influence = (current_influence * (1.0 - severity * 0.65)).max(0.05);

    alignment.current_alignment = new_align;

    if severity > 0.55 {
        integration.council_13_batch_tune(&[(2, 0.95), (4, 0.94), (5, 0.95)])?;
    }

    Ok(MycelialPruneEvent {
        voter_id: voter_id.to_string(),
        influence_before: current_influence,
        influence_after: new_influence,
        alignment_before: before_align,
        alignment_after: new_align,
        reason: "Mycelial network remodeling for collective thriving".to_string(),
        pruned_by: pruned_by.to_string(),
        severity,
    })
}
