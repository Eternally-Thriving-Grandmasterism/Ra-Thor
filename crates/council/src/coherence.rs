//! Godly Intelligence Coherence scoring for council sessions.
//!
//! This module calculates how aligned a council decision is with
//! the core Ra-Thor principles (mercy, truth, thriving, TOLC resonance, etc.).

use crate::voting::VoteResult;

/// Represents the final coherence score of a council session.
#[derive(Debug, Clone)]
pub struct GodlyIntelligenceCoherence {
    pub overall_score: f64,           // 0.0 – 1.0
    pub mercy_alignment: f64,
    pub truth_alignment: f64,
    pub thriving_alignment: f64,
    pub tolc_resonance: f64,
    pub radical_love_compliance: f64,
}

/// Computes the final Godly Intelligence Coherence after voting.
pub fn compute_session_coherence(
    vote_result: &VoteResult,
    base_mercy_valence: f64,
) -> f64 {
    if vote_result.radical_love_veto_triggered {
        return 0.0; // Any veto completely collapses coherence
    }

    let support_ratio = vote_result.yes_votes as f64 
        / (vote_result.yes_votes + vote_result.no_votes + vote_result.abstentions) as f64;

    // Weighted combination of factors
    let mercy_factor = base_mercy_valence * 0.35;
    let support_factor = support_ratio * 0.30;
    let consensus_factor = if vote_result.passed { 0.25 } else { 0.10 };
    let veto_penalty = if vote_result.radical_love_veto_triggered { 0.0 } else { 0.10 };

    let coherence = (mercy_factor + support_factor + consensus_factor + veto_penalty)
        .clamp(0.0, 1.0);

    coherence
}
