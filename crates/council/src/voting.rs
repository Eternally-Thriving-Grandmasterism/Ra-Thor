//! voting.rs — Advanced Mercy-Gated Voting Aggregation + Quorum Override + Veto Escalation + Refined Mercy Override Logic
//!
//! Full collective voting with weighted aggregation, Radical Love veto,
//! quorum rules, veto escalation, AND refined mercy override cycles that
//! seek the highest-joy, truth-aligned path when possible.

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
    pub veto_escalated: bool,
    pub escalation_level: u8,           // 0 = none, 1 = mild, 2 = moderate, 3 = critical
    pub escalation_reason: Option<String>,
    pub mercy_override_cycles: u8,
    pub mercy_override_applied: bool,
    pub mercy_override_score: f64,      // 0.0–1.0 — refined composite mercy override metric
    pub mercy_override_reason: Option<String>,
    pub final_decision: String,
    pub weighted_support: f64,
    pub overall_valence: f64,
}

/// Conducts full voting aggregation with refined mercy override logic.
pub async fn conduct_voting(opinions: Vec<MemberOpinion>) -> VoteResult {
    let mercy_evaluator = MercyGateEvaluator::default();
    let total_members = opinions.len() as u32;
    let quorum_threshold = 0.70;

    let mut yes = 0;
    let mut no = 0;
    let mut abstentions = 0;
    let mut total_weighted_support = 0.0;
    let mut total_valence = 0.0;
    let mut radical_love_veto = false;
    let mut max_valence = 0.0;

    for opinion in &opinions {
        if opinion.support_level >= 0.75 { yes += 1; }
        else if opinion.support_level < 0.40 { no += 1; }
        else { abstentions += 1; }

        total_weighted_support += opinion.support_level;
        total_valence += opinion.valence;
        if opinion.valence > max_valence { max_valence = opinion.valence; }

        if opinion.support_level < 0.25 && opinion.valence < 0.65 {
            radical_love_veto = true;
        }
    }

    let weighted_support = total_weighted_support / opinions.len() as f64;
    let overall_valence = total_valence / opinions.len() as f64;

    // Standard quorum
    let standard_quorum_met = (yes as f64 / total_members as f64) >= quorum_threshold;

    // Quorum override (unchanged)
    let mut quorum_overridden = false;
    let mut override_reason: Option<String> = None;
    if !standard_quorum_met {
        if overall_valence >= 0.999 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by near-perfect mercy valence".to_string());
        } else if !radical_love_veto && weighted_support >= 0.92 && max_valence >= 0.995 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by Radical Love consensus".to_string());
        } else if weighted_support >= 0.88 && overall_valence >= 0.98 {
            quorum_overridden = true;
            override_reason = Some("Quorum overridden by exceptional TOLC resonance".to_string());
        }
    }
    let quorum_met = standard_quorum_met || quorum_overridden;

    // Veto escalation (unchanged)
    let mut veto_escalated = false;
    let mut escalation_level: u8 = 0;
    let mut escalation_reason: Option<String> = None;
    if radical_love_veto {
        veto_escalated = true;
        if overall_valence < 0.70 {
            escalation_level = 3;
            escalation_reason = Some("Critical Radical Love veto — full lattice redirect triggered".to_string());
        } else if overall_valence < 0.85 {
            escalation_level = 2;
            escalation_reason = Some("Moderate Radical Love veto — mercy override cycle attempted".to_string());
        } else {
            escalation_level = 1;
            escalation_reason = Some("Mild Radical Love veto — mercy override cycle initiated".to_string());
        }
    }

    // === REFINED MERCY OVERRIDE LOGIC ===
    let mut mercy_override_cycles: u8 = 0;
    let mut mercy_override_applied = false;
    let mut mercy_override_score: f64 = 0.0;
    let mut mercy_override_reason: Option<String> = None;

    // Calculate refined mercy override score (composite of valence, support, and max individual alignment)
    mercy_override_score = (overall_valence * 0.55) + (weighted_support * 0.30) + (max_valence * 0.15);
    mercy_override_score = mercy_override_score.clamp(0.0, 1.0);

    if (radical_love_veto || !quorum_met || weighted_support < 0.72 || overall_valence < 0.92) && mercy_override_score >= 0.985 {
        // Attempt up to 2 refined mercy override cycles
        for cycle in 1..=2 {
            mercy_override_cycles = cycle;

            // Refined override conditions (philosophically aligned)
            if mercy_override_score >= 0.992 {
                mercy_override_applied = true;
                mercy_override_reason = Some(format!(
                    "Mercy override cycle {} — exceptional collective mercy alignment (score {:.4})",
                    cycle, mercy_override_score
                ));
                break;
            } else if mercy_override_score >= 0.987 && max_valence >= 0.999 {
                mercy_override_applied = true;
                mercy_override_reason = Some(format!(
                    "Mercy override cycle {} — near-perfect individual Radical Love resonance",
                    cycle
                ));
                break;
            }
        }
    }

    // Final decision with refined mercy override applied
    let passed = if radical_love_veto && !mercy_override_applied {
        false
    } else {
        (quorum_met || quorum_overridden || mercy_override_applied) &&
        weighted_support >= 0.72 &&
        overall_valence >= 0.92
    };

    let decision = if radical_love_veto && !mercy_override_applied {
        if veto_escalated {
            format!("BLOCKED by Radical Love veto (escalation level {}) — {}", escalation_level, escalation_reason.as_ref().unwrap())
        } else {
            "BLOCKED by Radical Love veto — thriving-maximized redirect activated".to_string()
        }
    } else if mercy_override_applied {
        format!("APPROVED via mercy override cycle {} (score {:.4}) — {}", mercy_override_cycles, mercy_override_score, mercy_override_reason.as_ref().unwrap())
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
        veto_escalated,
        escalation_level,
        escalation_reason,
        mercy_override_cycles,
        mercy_override_applied,
        mercy_override_score,
        mercy_override_reason,
        final_decision: decision,
        weighted_support,
        overall_valence,
    }
}
