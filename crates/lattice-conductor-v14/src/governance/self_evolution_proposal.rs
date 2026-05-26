//! Self-Evolution Proposal — First-class citizen in the governance cycle (Phase 14.1+)
//! Production-grade integration layer with rich mercy-alignment metadata.

use crate::governance::enhanced_exponential_conviction_staking::{ConvictionStake, score_self_evolution_proposal_with_mercy};
use crate::governance::mercy_weighted_quadratic_voting::{MercyWeightedVote, proposal_passes_mercy_quadratic};

/// A self-evolution proposal that participates in mercy-gated governance.
#[derive(Debug, Clone)]
pub struct SelfEvolutionProposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub proposed_by: String,
    pub mercy_alignment: f64,           // Overall mercy score from TOLC gates + councils
    pub conviction_stakes: Vec<ConvictionStake>,
    pub votes: Vec<MercyWeightedVote>,
    pub status: ProposalStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProposalStatus {
    Draft,
    UnderGovernanceReview,
    Passed,
    Rejected,
    Implemented,
}

impl SelfEvolutionProposal {
    pub fn new(id: String, title: String, description: String, proposed_by: String) -> Self {
        Self {
            id,
            title,
            description,
            proposed_by,
            mercy_alignment: 0.5, // Default neutral until evaluated
            conviction_stakes: Vec::new(),
            votes: Vec::new(),
            status: ProposalStatus::Draft,
        }
    }

    /// Attach mercy-aligned conviction stakes.
    pub fn add_conviction_stake(&mut self, stake: ConvictionStake) {
        self.conviction_stakes.push(stake);
    }

    /// Record a mercy-weighted vote.
    pub fn add_vote(&mut self, vote: MercyWeightedVote) {
        self.votes.push(vote);
    }

    /// Compute rich mercy metadata for this proposal.
    pub fn compute_mercy_metadata(&self) -> (f64, Vec<String>) {
        score_self_evolution_proposal_with_mercy(&self.id, &self.conviction_stakes)
    }

    /// Run the full governance evaluation.
    pub fn evaluate_governance(
        &self,
        threshold: f64,
    ) -> (bool, Vec<String>, f64) {
        let (passes, audit) = proposal_passes_mercy_quadratic(&self.votes, threshold);
        let (self_evo_score, _meta) = self.compute_mercy_metadata();

        let final_score = self_evo_score * if passes { 1.2 } else { 0.8 }; // Bonus for passing vote

        (passes, audit, final_score)
    }

    pub fn mark_under_review(&mut self) {
        self.status = ProposalStatus::UnderGovernanceReview;
    }

    pub fn finalize(&mut self, passed: bool) {
        self.status = if passed { ProposalStatus::Passed } else { ProposalStatus::Rejected };
    }
}