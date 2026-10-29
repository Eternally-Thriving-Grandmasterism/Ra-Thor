//! Count member votes.
//!
//! A session passes only when yes votes outnumber no votes and yes votes are
//! a strict majority of the roster. Admission-block reasons reject the
//! proposal even when every member voted yes.

use crate::deliberation::{MemberOpinion, MemberVote};

#[derive(Debug, Clone, PartialEq)]
pub struct VoteResult {
    pub passed: bool,
    pub yes_votes: u32,
    pub no_votes: u32,
    pub abstentions: u32,
    pub reasons: Vec<String>,
}

pub fn conduct_voting(opinions: &[MemberOpinion], blocks: &[String]) -> VoteResult {
    let mut yes_votes = 0;
    let mut no_votes = 0;
    let mut abstentions = 0;
    for opinion in opinions {
        match opinion.vote {
            MemberVote::Yes => yes_votes += 1,
            MemberVote::No => no_votes += 1,
            MemberVote::Abstain => abstentions += 1,
        }
    }

    if opinions.is_empty() {
        return VoteResult {
            passed: false,
            yes_votes,
            no_votes,
            abstentions,
            reasons: vec!["rejected: no members deliberated".to_string()],
        };
    }

    if !blocks.is_empty() {
        return VoteResult {
            passed: false,
            yes_votes,
            no_votes,
            abstentions,
            reasons: blocks.to_vec(),
        };
    }

    let majority = yes_votes > no_votes && (yes_votes as usize) * 2 > opinions.len();
    let reasons = if majority {
        Vec::new()
    } else {
        vec![format!(
            "rejected: {yes_votes} yes, {no_votes} no, {abstentions} abstain; majority of {} members did not vote yes",
            opinions.len()
        )]
    };

    VoteResult {
        passed: majority,
        yes_votes,
        no_votes,
        abstentions,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opinion(vote: MemberVote) -> MemberOpinion {
        MemberOpinion {
            member_id: 0,
            name: "member".to_string(),
            focus: "Truth".to_string(),
            support: match vote {
                MemberVote::Yes => 1.0,
                MemberVote::No => 0.0,
                MemberVote::Abstain => 0.25,
            },
            vote,
        }
    }

    #[test]
    fn two_yes_one_no_passes() {
        let opinions = vec![
            opinion(MemberVote::Yes),
            opinion(MemberVote::Yes),
            opinion(MemberVote::No),
        ];
        let result = conduct_voting(&opinions, &[]);
        assert!(result.passed);
        assert!(result.reasons.is_empty());
        assert_eq!(
            (result.yes_votes, result.no_votes, result.abstentions),
            (2, 1, 0)
        );
    }

    #[test]
    fn one_yes_two_no_fails() {
        let opinions = vec![
            opinion(MemberVote::Yes),
            opinion(MemberVote::No),
            opinion(MemberVote::No),
        ];
        let result = conduct_voting(&opinions, &[]);
        assert!(!result.passed);
        assert_eq!(result.yes_votes, 1);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("majority")));
    }

    #[test]
    fn admission_block_rejects_a_unanimous_yes() {
        let opinions = vec![
            opinion(MemberVote::Yes),
            opinion(MemberVote::Yes),
            opinion(MemberVote::Yes),
        ];
        let blocks = vec!["rejected: proposal tries to disable the cosmic loop".to_string()];
        let result = conduct_voting(&opinions, &blocks);
        assert!(!result.passed);
        assert_eq!(result.yes_votes, 3);
        assert_eq!(result.reasons, blocks);
    }
}
