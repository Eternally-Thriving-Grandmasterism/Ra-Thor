//! Self-Evolution Module (v14.8.2)
//! Higher-level abstractions for secure self-evolution loops.

use crate::governance::self_evolution_proposal::SelfEvolutionProposal;
use crate::LatticeConductorV14;

#[derive(Debug)]
pub struct SecureSubmissionResult {
    pub accepted: bool,
    pub audit: Vec<String>,
    pub score: f64,
}

/// Submit a self-evolution proposal securely.
/// Enforces Cosmic Loop via the conductor, then evaluates governance on the proposal.
pub fn submit_self_evolution_proposal_securely(
    conductor: &LatticeConductorV14,
    proposal: &mut SelfEvolutionProposal,
    signer_id: &str,
    threshold: f64,
) -> SecureSubmissionResult {
    // Non-bypassable Cosmic Loop enforcement before any governance action
    conductor.enforce_cosmic_loop_activation();
    conductor.arbitration_engine.before_council_arbitration();

    proposal.sign_with_post_quantum(signer_id);
    let (accepted, audit, score) = proposal.evaluate_governance(threshold);

    SecureSubmissionResult {
        accepted,
        audit,
        score,
    }
}

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

    pub fn advance(
        &mut self,
        conductor: &LatticeConductorV14,
        signer_id: &str,
        threshold: f64,
    ) -> SecureSubmissionResult {
        self.iteration += 1;
        self.state = EvolutionLoopState::UnderGovernance;

        let result =
            submit_self_evolution_proposal_securely(conductor, &mut self.proposal, signer_id, threshold);

        self.state = if result.accepted {
            EvolutionLoopState::Accepted
        } else {
            EvolutionLoopState::Rejected
        };

        result
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            EvolutionLoopState::Initialized | EvolutionLoopState::UnderGovernance
        )
    }
}
