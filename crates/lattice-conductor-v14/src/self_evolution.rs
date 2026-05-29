//! Self-Evolution Module
//! Higher-level abstractions for secure self-evolution loops.

use crate::governance::self_evolution_proposal::SelfEvolutionProposal;
use crate::LatticeConductorV14;

/// Result of a secure self-evolution proposal submission.
#[derive(Debug)]
pub struct SecureSubmissionResult {
    pub accepted: bool,
    pub audit: Vec<String>,
    pub score: f64,
}

/// Submit a self-evolution proposal securely.
pub fn submit_self_evolution_proposal_securely(
    conductor: &LatticeConductorV14,
    proposal: &mut SelfEvolutionProposal,
    signer_id: &str,
    threshold: f64,
) -> SecureSubmissionResult {
    proposal.sign_with_post_quantum(signer_id);
    let (accepted, audit, score) = conductor.submit_secure_governance_proposal(proposal, threshold);

    SecureSubmissionResult { accepted, audit, score }
}

// ==================== Looping Mechanics ====================

#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionLoopState {
    Initialized,
    UnderGovernance,
    Accepted,
    Rejected,
}

#[derive(Debug)]
pub struct SelfEvolutionLoop {
    pub proposal: SelfEvolutionProposal,
    pub state: EvolutionLoopState,
    pub iteration: u32,
}

impl SelfEvolutionLoop {
    pub fn new(proposal: SelfEvolutionProposal) -> Self {
        Self {
            proposal,
            state: EvolutionLoopState::Initialized,
            iteration: 0,
        }
    }

    /// Advance one secure iteration of the evolution loop.
    pub fn advance(
        &mut self,
        conductor: &LatticeConductorV14,
        signer_id: &str,
        threshold: f64,
    ) -> SecureSubmissionResult {
        self.iteration += 1;
        self.state = EvolutionLoopState::UnderGovernance;

        let result = submit_self_evolution_proposal_securely(
            conductor,
            &mut self.proposal,
            signer_id,
            threshold,
        );

        self.state = if result.accepted {
            EvolutionLoopState::Accepted
        } else {
            EvolutionLoopState::Rejected
        };

        result
    }

    pub fn is_active(&self) -> bool {
        matches!(self.state, EvolutionLoopState::Initialized | EvolutionLoopState::UnderGovernance)
    }
}