//! deliberation.rs — Advanced Mercy-Gated Parallel Deliberation with Revised Weighting
//!
//! Each council member reasons individually in parallel, with full
//! mercy gate evaluation, rich TOLC affinity, and revised weighting logic.

use crate::council_session::CouncilProposal;
use crate::member_profiles::CouncilMemberProfile;
use crate::tolc::calculate_tolc_resonance;
use ra_thor_mercy::MercyGateEvaluator;
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberOpinion {
    pub member_id: u32,
    pub name: String,
    pub support_level: f64,        // Final weighted mercy + TOLC score
    pub reasoning: String,
    pub concerns: Vec<String>,
    pub valence: f64,
    pub tolc_resonance: f64,
    pub tolc_order: u32,
}

/// Runs full parallel deliberation using rich member profiles and revised weighting
pub async fn run_parallel_deliberation(
    profiles: &[CouncilMemberProfile],
    proposal: &CouncilProposal,
) -> Vec<MemberOpinion> {
    let mercy_evaluator = MercyGateEvaluator::default();
    let mut opinions = Vec::with_capacity(profiles.len());
    let mut rng = rand::thread_rng();

    for profile in profiles {
        let tolc_resonance = calculate_tolc_resonance(
            profile,
            proposal.complexity,
            proposal.impact_level,
        );

        let base_support = 0.70 + rng.gen_range(-0.22..0.28);

        // Mercy gate evaluation on this member's view
        let raw_reasoning = generate_member_reasoning(&profile.member, proposal, base_support);
        let mercy_valence = mercy_evaluator.evaluate(&raw_reasoning);

        // === REVISED WEIGHTING LOGIC ===
        // Mercy valence has highest priority (core principle)
        // TOLC resonance provides strong mathematical harmony
        // Member's personal mercy_bias adds individual character
        let final_support = (
            base_support          * 0.40 +
            mercy_valence         * 0.35 +
            tolc_resonance        * 0.20 +
            profile.mercy_bias    * 0.05
        )
        .clamp(0.0, 1.0);

        let concerns = generate_member_concerns(&profile.member, proposal, final_support);

        opinions.push(MemberOpinion {
            member_id: profile.member.id,
            name: profile.member.name.clone(),
            support_level: final_support,
            reasoning: raw_reasoning,
            concerns,
            valence: mercy_valence,
            tolc_resonance,
            tolc_order: profile.tolc_affinity.derivative_order,
        });
    }

    opinions
}

fn generate_member_reasoning(
    member: &patsagi_councils::CouncilMember,
    proposal: &CouncilProposal,
    support: f64,
) -> String {
    if support > 0.85 {
        format!(
            "{} strongly supports this proposal. It aligns perfectly with long-term thriving, mercy principles, and high TOLC resonance.",
            member.name
        )
    } else if support > 0.65 {
        format!(
            "{} sees strong value in the proposal but recommends additional mercy gate monitoring and deeper TOLC resonance validation.",
            member.name
        )
    } else {
        format!(
            "{} has significant reservations. The proposal requires stronger safeguards for Radical Love and TOLC alignment before full support.",
            member.name
        )
    }
}

fn generate_member_concerns(
    member: &patsagi_councils::CouncilMember,
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
