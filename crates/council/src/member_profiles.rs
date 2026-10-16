//! Standing council roster used by a session.
//!
//! Each profile carries lowercase tokens. [`crate::tolc::calculate_tolc_resonance`]
//! counts how many of those tokens appear in the proposal.

/// One member of the standing roster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CouncilMemberProfile {
    pub id: u32,
    pub name: &'static str,
    /// One public TOLC name this member weighs.
    pub focus: &'static str,
    /// Lowercase substrings. Resonance is hits divided by this length.
    pub tokens: &'static [&'static str],
}

/// The three members a session always deliberates with.
pub fn standing_profiles() -> &'static [CouncilMemberProfile] {
    &[
        CouncilMemberProfile {
            id: 1,
            name: "Truth anchor",
            focus: "Truth",
            tokens: &["truth", "record", "human"],
        },
        CouncilMemberProfile {
            id: 2,
            name: "Order anchor",
            focus: "Order",
            tokens: &["order", "layer 0", "gate"],
        },
        CouncilMemberProfile {
            id: 3,
            name: "Service anchor",
            focus: "Service",
            tokens: &["service", "research", "draft"],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roster_tokens_are_lowercase_and_non_empty() {
        let profiles = standing_profiles();
        assert_eq!(profiles.len(), 3);
        for profile in profiles {
            assert!(!profile.tokens.is_empty(), "{}", profile.name);
            for token in profile.tokens {
                assert_eq!(token, &token.to_ascii_lowercase(), "{token}");
            }
        }
    }
}
