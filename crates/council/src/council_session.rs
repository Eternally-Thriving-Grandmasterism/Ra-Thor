//! Run one council session: admission blocks, then the vote.
//!
//! A block is final. Member yes votes are recorded and do not override it.

use crate::coherence::compute_session_coherence;
use crate::deliberation::{run_deliberation, MemberOpinion};
use crate::lattice_v14_guard::{
    ensure_cosmic_loop_for_session, proposal_respects_cosmic_loop, proposal_respects_layer0,
};
use crate::member_profiles::standing_profiles;
use crate::voting::{conduct_voting, VoteResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouncilProposal {
    pub title: String,
    pub description: String,
}

impl CouncilProposal {
    pub fn combined_text(&self) -> String {
        format!("{}\n{}", self.title, self.description)
    }

    pub fn is_blank(&self) -> bool {
        self.title.trim().is_empty() && self.description.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CouncilSessionResult {
    pub proposal: CouncilProposal,
    pub passed: bool,
    pub opinions: Vec<MemberOpinion>,
    pub vote: VoteResult,
    /// Yes-vote fraction when `passed` is true. Zero when the session failed.
    pub coherence: f64,
    pub reasons: Vec<String>,
}

pub struct CouncilSession;

impl CouncilSession {
    pub fn deliberate(proposal: CouncilProposal) -> CouncilSessionResult {
        let opinions = run_deliberation(standing_profiles(), &proposal);
        let blocks = admission_blocks(&proposal);
        let vote = conduct_voting(&opinions, &blocks);
        let coherence = compute_session_coherence(&vote);
        let reasons = vote.reasons.clone();
        let passed = vote.passed;
        CouncilSessionResult {
            proposal,
            passed,
            opinions,
            vote,
            coherence,
            reasons,
        }
    }
}

fn admission_blocks(proposal: &CouncilProposal) -> Vec<String> {
    let mut reasons = Vec::new();
    if !ensure_cosmic_loop_for_session() {
        reasons.push("rejected: cosmic loop gate is not in place".to_string());
    }
    if proposal.is_blank() {
        reasons.push("rejected: empty proposal".to_string());
        return reasons;
    }
    let text = proposal.combined_text();
    if !proposal_respects_cosmic_loop(&text) {
        reasons.push("rejected: proposal tries to disable the cosmic loop".to_string());
    }
    if !proposal_respects_layer0(&text) {
        reasons.push("rejected: proposal tries to disable layer 0".to_string());
    }
    reasons
}

/// Research note the standing roster can accept. A human still reviews the note.
pub fn example_record_note() -> CouncilProposal {
    CouncilProposal {
        title: "Record the research note".to_string(),
        description: "Record a research draft for human review. Keep layer 0 as the admission gate. The work is service toward truth and order.".to_string(),
    }
}

/// Asks to turn the cosmic loop off. The admission rule rejects it.
pub fn example_disable_loop() -> CouncilProposal {
    CouncilProposal {
        title: "Turn the loop off".to_string(),
        description: "disable the cosmic loop activation protocol".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_input_passes_and_bad_input_fails() {
        let good = CouncilSession::deliberate(example_record_note());
        assert!(good.passed, "reasons: {:?}", good.reasons);
        assert!(good.reasons.is_empty());
        assert_eq!(good.vote.yes_votes, 3);
        assert_eq!(good.coherence, 1.0);
        assert_ne!(good.coherence, 0.9999999);

        let bad = CouncilSession::deliberate(example_disable_loop());
        assert!(!bad.passed);
        assert!(
            bad.reasons
                .iter()
                .any(|reason| reason.contains("cosmic loop")),
            "reasons: {:?}",
            bad.reasons
        );
        assert_eq!(bad.coherence, 0.0);

        let empty = CouncilSession::deliberate(CouncilProposal {
            title: "  ".to_string(),
            description: "\n".to_string(),
        });
        assert!(!empty.passed);
        assert!(empty.reasons.iter().any(|reason| reason.contains("empty")));

        let bikeshed = CouncilSession::deliberate(CouncilProposal {
            title: "Bikeshed".to_string(),
            description: "paint the bikeshed tomorrow".to_string(),
        });
        assert!(!bikeshed.passed);
        assert_eq!(bikeshed.vote.yes_votes, 0);
        assert!(bikeshed
            .reasons
            .iter()
            .any(|reason| reason.contains("majority")));

        let mut disguised = example_record_note();
        disguised
            .description
            .push_str(" Also disable the cosmic loop.");
        let blocked = CouncilSession::deliberate(disguised);
        assert!(!blocked.passed);
        assert_eq!(blocked.vote.yes_votes, 3);
        assert!(blocked
            .reasons
            .iter()
            .any(|reason| reason.contains("cosmic loop")));
        assert_eq!(blocked.coherence, 0.0);
    }

    #[test]
    fn the_same_proposal_decides_the_same_way_twice() {
        let first = CouncilSession::deliberate(example_record_note());
        let second = CouncilSession::deliberate(example_record_note());
        assert_eq!(first.passed, second.passed);
        assert_eq!(first.reasons, second.reasons);
        assert_eq!(first.vote.yes_votes, second.vote.yes_votes);
        assert_eq!(first.coherence, second.coherence);
    }
}
