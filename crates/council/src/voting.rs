//! voting.rs — Advanced Mercy-Gated Voting Aggregation + Quorum Override + Veto Escalation Paths
//!
//! Full collective voting with weighted aggregation, Radical Love veto,
//! quorum rules, mercy override cycles, AND distinct veto escalation paths.

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
    pub escalation_path: String,        // Clear description of the path taken
    pub escalation_reason: Option<String>,
    pub mercy_override_cycles: u8,
    pub mercy_override_applied: bool,
    pub mercy_override_score: f64,
    pub mercy_override_reason: Option<String>,
    pub final_decision: String,
    pub weighted_support: f64,
    pub overall_valence: f64,
}

/// Conducts full voting aggregation with distinct veto escalation paths.
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

    // Standard quorum + override (unchanged)
    let standard_quorum_met = (yes as f64 / total_members as f64) >= quorum_threshold;
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

    // === VETO ESCALATION PATHS ===
    let mut veto_escalated = false;
    let mut escalation_level: u8 = 0;
    let mut escalation_path = "none".to_string();
    let mut escalation_reason: Option<String> = None;

    if radical_love_veto {
        veto_escalated = true;

        if overall_valence < 0.70 {
            // Path 3: Critical — full lattice redirect
            escalation_level = 3;
            escalation_path = "critical-lattice-redirect".to_string();
            escalation_reason = Some("Critical Radical Love veto — full lattice redirect to highest-joy path".to_string());
        } else if overall_valence < 0.85 {
            // Path 2: Moderate — mercy override cycle + re-deliberation
            escalation_level = 2;
            escalation_path = "moderate-mercy-override".to_string();
            escalation_reason = Some("Moderate Radical Love veto — mercy override cycle + re-deliberation initiated".to_string());
        } else {
            // Path 1: Mild — gentle re-evaluation
            escalation_level = 1;
            escalation_path = "mild-re-evaluation".to_string();
            escalation_reason = Some("Mild Radical Love veto — gentle re-evaluation with mercy guidance".to_string());
        }
    }

    // Mercy override cycles (refined, unchanged)
    let mut mercy_override_cycles: u8 = 0;
    let mut mercy_override_applied = false;
    let mut mercy_override_score: f64 = (overall_valence * 0.55) + (weighted_support * 0.30) + (max_valence * 0.15);
    mercy_override_score = mercy_override_score.clamp(0.0, 1.0);
    let mut mercy_override_reason: Option<String> = None;

    if (radical_love_veto || !quorum_met || weighted_support < 0.72 || overall_valence < 0.92) && mercy_override_score >= 0.985 {
        for cycle in 1..=2 {
            mercy_override_cycles = cycle;
            if mercy_override_score >= 0.992 {
                mercy_override_applied = true;
                mercy_override_reason = Some(format!("Mercy override cycle {} — exceptional collective mercy alignment (score {:.4})", cycle, mercy_override_score));
                break;
            } else if mercy_override_score >= 0.987 && max_valence >= 0.999 {
                mercy_override_applied = true;
                mercy_override_reason = Some(format!("Mercy override cycle {} — near-perfect individual Radical Love resonance", cycle));
                break;
            }
        }
    }

    // Final decision with escalation paths applied
    let passed = if radical_love_veto && !mercy_override_applied {
        false
    } else {
        (quorum_met || quorum_overridden || mercy_override_applied) &&
        weighted_support >= 0.72 &&
        overall_valence >= 0.92
    };

    let decision = match (radical_love_veto, mercy_override_applied, escalation_path.as_str()) {
        (true, false, "critical-lattice-redirect") => "BLOCKED by Critical Radical Love veto — full lattice redirect to highest-joy path".to_string(),
        (true, false, "moderate-mercy-override") => "BLOCKED by Moderate Radical Love veto — mercy override cycle attempted".to_string(),
        (true, false, "mild-re-evaluation") => "BLOCKED by Mild Radical Love veto — gentle re-evaluation initiated".to_string(),
        (true, true, _) => format!("APPROVED via mercy override after Radical Love veto (escalation level {})", escalation_level),
        (false, true, _) => format!("APPROVED via mercy override cycle {} (score {:.4})", mercy_override_cycles, mercy_override_score),
        (false, false, _) if passed && quorum_overridden => "APPROVED with mercy-gated quorum override".to_string(),
        (false, false, _) if passed => "APPROVED with full mercy-aligned consensus".to_string(),
        _ => "NOT APPROVED — insufficient support or coherence".to_string(),
    };

    // Final mercy gate
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
        escalation_path,
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
