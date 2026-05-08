//! voting.rs — Advanced Mercy-Gated Voting Aggregation Logic
//!
//! This module performs collective voting with full Radical Love veto,
//! weighted aggregation, quorum requirements, and final mercy coherence check.

use crate::deliberation::MemberOpinion;
use ra_thor_mercy::MercyGateEvaluator;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResult {
    pub passed: bool,
    pub yes_votes: u32,
    pub no_votes: u32,
    pub abstentions: u32,
    pub radical_love_veto_triggered: bool,
    pub final_decision: String,
    pub weighted_support: f64,
    pub overall_valence: f64,
}

/// Conducts full voting aggregation with Radical Love veto and mercy weighting.
pub async fn conduct_voting(opinions: Vec<MemberOpinion>) -> VoteResult {
    let mercy_evaluator = MercyGateEvaluator::default();
    let total_members = opinions.len() as u32;

    let mut yes = 0;
    let mut no = 0;
    let mut abstentions = 0;
    let mut total_weighted_support = 0.0;
    let mut total_valence = 0.0;
    let mut radical_love_veto = false;

    for opinion in &opinions {
        // Aggregate support
        if opinion.support_level >= 0.75 {
            yes += 1;
        } else if opinion.support_level < 0.40 {
            no += 1;
        } else {
            abstentions += 1;
        }

        total_weighted_support += opinion.support_level;
        total_valence += opinion.valence;

        // Radical Love veto check (highest priority override)
        if opinion.support_level < 0.25 && opinion.valence < 0.65 {
            radical_love_veto = true;
        }
    }

    let weighted_support = total_weighted_support / opinions.len() as f64;
    let overall_valence = total_valence / opinions.len() as f64;

    // Final decision logic with mercy gates
    let passed = if radical_love_veto {
        false
    } else {
        // Quorum + strong majority threshold
        let majority = yes as f64 / total_members as f64 >= 0.60;
        let high_coherence = weighted_support >= 0.72 && overall_valence >= 0.92;
        majority && high_coherence
    };

    let decision = if radical_love_veto {
        "BLOCKED by Radical Love veto — thriving-maximized redirect activated".to_string()
    } else if passed {
        "APPROVED with full mercy-aligned consensus".to_string()
    } else {
        "NOT APPROVED — insufficient weighted support or coherence".to_string()
    };

    // Final mercy gate on the entire vote
    let final_mercy_check = mercy_evaluator.evaluate(&decision);
    let final_passed = passed && final_mercy_check >= 0.92;

    VoteResult {
        passed: final_passed,
        yes_votes: yes,
        no_votes: no,
        abstentions,
        radical_love_veto_triggered: radical_love_veto,
        final_decision: decision,
        weighted_support,
        overall_valence,
    }
}
