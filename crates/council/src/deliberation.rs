//! One opinion per standing member.
//!
//! Support is token overlap with the proposal text. A member votes yes when
//! at least half of their tokens appear, no when none appear, and abstains
//! in between. There is no random term.

use crate::council_session::CouncilProposal;
use crate::member_profiles::CouncilMemberProfile;
use crate::tolc::calculate_tolc_resonance;

/// A member votes yes at or above this fraction of their own tokens.
pub const YES_SUPPORT: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberVote {
    Yes,
    No,
    Abstain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemberOpinion {
    pub member_id: u32,
    pub name: String,
    pub focus: String,
    /// Fraction of this member's tokens found in the proposal. Range 0 to 1.
    pub support: f64,
    pub vote: MemberVote,
}

pub fn run_deliberation(
    profiles: &[CouncilMemberProfile],
    proposal: &CouncilProposal,
) -> Vec<MemberOpinion> {
    let text = proposal.combined_text();
    profiles
        .iter()
        .map(|profile| {
            let support = calculate_tolc_resonance(profile.tokens, &text);
            MemberOpinion {
                member_id: profile.id,
                name: profile.name.to_string(),
                focus: profile.focus.to_string(),
                support,
                vote: vote_from_support(support),
            }
        })
        .collect()
}

fn vote_from_support(support: f64) -> MemberVote {
    if support >= YES_SUPPORT {
        MemberVote::Yes
    } else if support == 0.0 {
        MemberVote::No
    } else {
        MemberVote::Abstain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_the_tokens_is_yes_and_one_hit_of_three_abstains() {
        assert_eq!(vote_from_support(0.5), MemberVote::Yes);
        assert_eq!(vote_from_support(1.0 / 3.0), MemberVote::Abstain);
        assert_eq!(vote_from_support(0.0), MemberVote::No);
    }
}
