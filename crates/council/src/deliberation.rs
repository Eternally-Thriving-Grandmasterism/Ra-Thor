//! deliberation.rs — Advanced Mercy-Gated Parallel Deliberation
//!
//! This is the heart of the PATSAGi Council Simulator.
//! Each council member reasons individually in parallel, with full
//! mercy gate evaluation, TOLC resonance weighting, and valence scoring.

use crate::council_session::CouncilProposal;
use patsagi_councils::CouncilMember;
use ra_thor_mercy::MercyGateEvaluator;
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberOpinion {
    pub member_id: u32,
    pub name: String,
    pub support_level: f64,        // 0.0 – 1.0 (final mercy-gated score)
    pub reasoning: String,
    pub concerns: Vec<String>,
    pub valence: f64,
    pub tolc_resonance: f64,
}

/// Runs full parallel deliberation across all council members
/// with mercy gates and TOLC resonance applied to every opinion.
pub async fn run_parallel_deliberation(
    members: &[CouncilMember],
    proposal: &CouncilProposal,
) -> Vec<MemberOpinion> {
    let mercy_evaluator = MercyGateEvaluator::default();
    let mut opinions = Vec::with_capacity(members.len());
    let mut rng = rand::thread_rng();

    // Parallel deliberation (in real production this would use rayon or tokio tasks)
    for member in members {
        let base_support = 0.70 + rng.gen_range(-0.25..0.35);
        let tolc_resonance = calculate_tolc_resonance(proposal, member);

        // Mercy gate evaluation on this member's view
        let raw_reasoning = generate_member_reasoning(member, proposal, base_support);
        let mercy_valence = mercy_evaluator.evaluate(&raw_reasoning);

        let final_support = (base_support * 0.6 + mercy_valence * 0.4).clamp(0.0, 1.0);

        let concerns = generate_member_concerns(member, proposal, final_support);

        opinions.push(MemberOpinion {
            member_id: member.id,
            name: member.name.clone(),
            support_level: final_support,
            reasoning: raw_reasoning,
            concerns,
            valence: mercy_valence,
            tolc_resonance,
        });
    }

    opinions
}

fn calculate_tolc_resonance(proposal: &CouncilProposal, member: &CouncilMember) -> f64 {
    // Simple but meaningful resonance based on proposal complexity and member expertise
    let complexity_factor = proposal.complexity.clamp(0.0, 1.0);
    let expertise_bonus = if member.expertise.contains("TOLC") || member.expertise.contains("Truth") {
        0.25
    } else if member.expertise.contains("Mercy") || member.expertise.contains("Love") {
        0.15
    } else {
        0.0
    };
    (0.75 + complexity_factor * 0.2 + expertise_bonus).clamp(0.6, 1.0)
}

fn generate_member_reasoning(
    member: &CouncilMember,
    proposal: &CouncilProposal,
    support: f64,
) -> String {
    if support > 0.85 {
        format!(
            "{} strongly supports this proposal. It aligns perfectly with long-term thriving, mercy principles, and TOLC resonance.",
            member.name
        )
    } else if support > 0.65 {
        format!(
            "{} sees strong value in the proposal but recommends additional mercy gate monitoring and TOLC validation.",
            member.name
        )
    } else {
        format!(
            "{} has significant reservations. The proposal requires stronger safeguards for Radical Love and sovereignty before full support.",
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

    if support < 0.85 {
        concerns.push("Potential short-term disruption to Powrush RBE economy balance".to_string());
    }
    if proposal.complexity > 0.75 {
        concerns.push("High complexity requires deeper TOLC resonance analysis before final approval".to_string());
    }
    if support < 0.65 {
        concerns.push("Strong request for explicit Radical Love veto safeguards".to_string());
    }
    if member.expertise.contains("Sovereignty") && support < 0.8 {
        concerns.push("Potential impact on individual autonomy must be re-evaluated".to_string());
    }

    concerns
}
