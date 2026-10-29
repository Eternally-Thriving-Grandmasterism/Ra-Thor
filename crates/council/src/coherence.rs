//! Yes-vote fraction for a session that passed.
//!
//! A rejected session scores 0. The number is a vote fraction, not a valence.

use crate::voting::VoteResult;

pub fn compute_session_coherence(vote: &VoteResult) -> f64 {
    let total = vote.yes_votes + vote.no_votes + vote.abstentions;
    if !vote.passed || total == 0 {
        return 0.0;
    }
    vote.yes_votes as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vote(passed: bool, yes: u32, no: u32, abstain: u32) -> VoteResult {
        VoteResult {
            passed,
            yes_votes: yes,
            no_votes: no,
            abstentions: abstain,
            reasons: Vec::new(),
        }
    }

    #[test]
    fn passed_session_is_the_yes_fraction() {
        assert_eq!(compute_session_coherence(&vote(true, 3, 0, 0)), 1.0);
        assert_eq!(compute_session_coherence(&vote(true, 2, 1, 0)), 2.0 / 3.0);
        assert_eq!(compute_session_coherence(&vote(false, 3, 0, 0)), 0.0);
        assert_ne!(compute_session_coherence(&vote(true, 3, 0, 0)), 0.9999999);
    }
}
