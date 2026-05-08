//! voting.rs — Advanced Mercy-Gated Voting Aggregation + Quorum Override Logic
//!
//! Full collective voting with weighted aggregation, Radical Love veto,
//! standard quorum requirement, AND mercy-gated quorum override rules.

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
    pub quorum_met: bool,
    pub quorum_overridden: bool,
    pub override_reason: Option<String>,
    pub final_decision: String,
    pub weighted_support: f64,
    pub overall_valence: f64,
}

/// Conducts full voting aggregation with Radical Love veto,
/// standard quorum, and mercy-gated quorum override logic.
pub async fn conduct_voting(opinions: Vec<MemberOpinion>) -> VoteResult {
    let mercy_evaluator = MercyGateEvaluator::default();
    let total_members = opinions.len() as u32;
    let quorum_threshold = 0.70; // 70% standard quorum

    let mut yes = 0;
    let mut no = 0;
    let mut abstentions = 0;
    let mut total_weighted_support = 0.0;
    let mut total_valence = 0.0;
    let mut radical_love_veto = false;
    let mut max_valence = 0.0;

    for opinion in &opinions {
        if opinion.support_level >= 0.75 {
            yes += 1;
        } else if opinion.support_level < 0.40 {
            no += 1;
        } else {
            abstentions += 1;
        }

        total_weighted_support += opinion.support_level;
        total_valence += opinion.valence;
        if opinion.valence > max_valence {
            max_valence = opinion.valence;
        }

        // Radical Love veto (absolute highest priority)
        if opinion.support_level < 0.25 && opinion.valence < 0.65 {
            radical_love_veto = true;
        }
    }

    let weighted_support = total_weighted_support / opinions.len() as f64;
    let overall_valence = total_valence / opinions.len() as f64;

    // === STANDARD QUORUM CALCULATION ===
    let standard_quorum_met = (yes as f64 / total_members as f64) >= quorum_threshold;

    // === QUORUM OVERRIDE LOGIC (Mercy-Gated) ===
    let mut quorum_overridden = false;
    let mut override_reason: Option<String> = None;

    if !standard_quorum_met {
        // Override Condition 1: Perfect or near-perfect overall valence
        if overall_valence >= 0.999 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by near-perfect mercy valence (≥0.999)".to_string());
        }
        // Override Condition 2: Radical Love consensus (no veto + extremely high support)
        else if !radical_love_veto && weighted_support >= 0.92 && max_valence >= 0.995 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by Radical Love consensus".to_string());
        }
        // Override Condition 3: Exceptional TOLC resonance (inferred from high coherence)
        else if weighted_support >= 0.88 && overall_valence >= 0.98 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by exceptional TOLC resonance".to_string());
        }
    }

    let quorum_met = standard_quorum_met || quorum_overridden;

    // Final decision logic
    let passed = if radical_love_veto {
        false
    } else {
        quorum_met && weighted_support >= 0.72 && overall_valence >= 0.92
    };

    let decision = if radical_love_veto {
        "BLOCKED by Radical Love veto — thriving-maximized redirect activated".to_string()
    } else if passed {
        if quorum_overridden {
            "APPROVED with mercy-gated quorum override".to_string()
        } else {
            "APPROVED with full mercy-aligned consensus".to_string()
        }
    } else {
        "NOT APPROVED — insufficient support or coherence".to_string()
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
        quorum_met,
        quorum_overridden,
        override_reason,
        final_decision: decision,
        weighted_support,
        overall_valence,
    }
}
