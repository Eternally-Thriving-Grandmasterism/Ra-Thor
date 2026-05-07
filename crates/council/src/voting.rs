//! Voting and consensus logic for the PATSAGi Councils.
//!
//! This module handles collective voting among council members,
//! including the critical Radical Love veto mechanism.

use crate::deliberation::MemberOpinion;

use serde::{Serialize, Deserialize};

/// Result of a council voting round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResult {
    pub passed: bool,
    pub yes_votes: u32,
    pub no_votes: u32,
    pub abstentions: u32,
    pub radical_love_veto_triggered: bool,
    pub final_decision: String,
}

/// Conducts voting among council members with Radical Love veto support.
pub async fn conduct_voting(opinions: Vec<MemberOpinion>) -> VoteResult {
    let total = opinions.len() as u32;
    let mut yes = 0;
    let mut no = 0;
    let mut abstentions = 0;

    let mut radical_love_veto = false;

    for opinion in &opinions {
        if opinion.support_level >= 0.75 {
            yes += 1;
        } else if opinion.support_level < 0.4 {
            no += 1;
        } else {
            abstentions += 1;
        }

        // Radical Love veto: if any member has very strong opposition + low valence
        if opinion.support_level < 0.3 && opinion.valence < 0.6 {
            radical_love_veto = true;
        }
    }

    let passed = if radical_love_veto {
        false
    } else {
        yes as f64 / total as f64 >= 0.6   // Simple majority with strong threshold
    };

    let decision = if radical_love_veto {
        "BLOCKED by Radical Love veto".to_string()
    } else if passed {
        "APPROVED with mercy-aligned consensus".to_string()
    } else {
        "NOT APPROVED — insufficient support".to_string()
    };

    VoteResult {
        passed,
        yes_votes: yes,
        no_votes: no,
        abstentions,
        radical_love_veto_triggered: radical_love_veto,
        final_decision: decision,
    }
}
