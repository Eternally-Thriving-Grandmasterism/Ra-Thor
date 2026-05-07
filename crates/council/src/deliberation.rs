//! Parallel deliberation logic for the PATSAGi Councils.
//!
//! This module handles how the 13+ Council Members reason in parallel
//! about a proposal before voting.

use crate::council_session::CouncilProposal;
use patsagi_councils::CouncilMember;

use serde::{Serialize, Deserialize};
use rand::Rng;

/// Represents one council member's opinion after deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberOpinion {
    pub member_id: u32,
    pub name: String,
    pub support_level: f64,        // 0.0 – 1.0
    pub reasoning: String,
    pub concerns: Vec<String>,
    pub valence: f64,
}

/// Runs parallel deliberation among all council members.
pub async fn run_parallel_deliberation(
    members: &[CouncilMember],
    proposal: &CouncilProposal,
) -> Vec<MemberOpinion> {
    let mut opinions = Vec::with_capacity(members.len());
    let mut rng = rand::thread_rng();

    for member in members {
        // Simulate individual member reasoning (in real version this would be much deeper)
        let base_support = 0.75 + rng.gen_range(-0.15..0.20);
        let support_level = base_support.clamp(0.0, 1.0);

        let reasoning = generate_member_reasoning(member, proposal, support_level);
        let concerns = generate_member_concerns(member, proposal, support_level);

        let valence = (support_level * 0.7 + 0.3).clamp(0.6, 0.999);

        opinions.push(MemberOpinion {
            member_id: member.id,
            name: member.name.clone(),
            support_level,
            reasoning,
            concerns,
            valence,
        });
    }

    opinions
}

fn generate_member_reasoning(
    member: &CouncilMember,
    proposal: &CouncilProposal,
    support: f64,
) -> String {
    if support > 0.85 {
        format!(
            "{} strongly supports this proposal. It aligns well with long-term thriving and mercy principles.",
            member.name
        )
    } else if support > 0.65 {
        format!(
            "{} sees value in the proposal but notes it requires careful monitoring of mercy alignment.",
            member.name
        )
    } else {
        format!(
            "{} has reservations. The proposal needs stronger safeguards before full support.",
            member.name
        )
    }
}

fn generate_member_concerns(
    member: &CouncilMember,
    proposal: &CouncilProposal,
    support: f64,
) -> Vec<String> {
    let mut concerns = vec![];

    if support < 0.8 {
        concerns.push("Potential impact on current Powrush economy balance".to_string());
    }
    if proposal.complexity > 0.8 {
        concerns.push("High complexity may require additional TOLC resonance checks".to_string());
    }
    if support < 0.7 {
        concerns.push("Request for stronger Radical Love safeguards".to_string());
    }

    concerns
}
