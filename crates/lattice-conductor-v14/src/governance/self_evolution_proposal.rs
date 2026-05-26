//! Self-Evolution Proposal with Post-Quantum Signature Support

use crate::post_quantum_signatures::{PostQuantumSignature, create_post_quantum_signature};

#[derive(Debug, Clone)]
pub struct SelfEvolutionProposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub proposed_by: String,
    pub mercy_alignment: f64,
    pub conviction_stakes: Vec<crate::governance::enhanced_exponential_conviction_staking::ConvictionStake>,
    pub votes: Vec<crate::governance::mercy_weighted_quadratic_voting::MercyWeightedVote>,
    pub status: ProposalStatus,
    pub pq_signature: Option<PostQuantumSignature>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProposalStatus { Draft, UnderGovernanceReview, Passed, Rejected, Implemented }

impl SelfEvolutionProposal {
    pub fn new(id: String, title: String, description: String, proposed_by: String) -> Self {
        Self {
            id, title, description, proposed_by,
            mercy_alignment: 0.5,
            conviction_stakes: vec![],
            votes: vec![],
            status: ProposalStatus::Draft,
            pq_signature: None,
        }
    }

    pub fn sign_with_post_quantum(&mut self, signer_id: &str) {
        let message = format!("proposal:{}:{}", self.id, self.title).into_bytes();
        self.pq_signature = Some(create_post_quantum_signature(signer_id, &message));
    }

    pub fn has_valid_pq_signature(&self) -> bool {
        self.pq_signature.is_some()
    }
}