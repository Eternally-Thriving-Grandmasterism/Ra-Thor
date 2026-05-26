//! Self-Evolution Module
//! Provides higher-level abstractions for secure self-evolution loops.

use crate::governance::self_evolution_proposal::SelfEvolutionProposal;
use crate::LatticeConductorV14;

/// A secure self-evolution proposal submission result.
#[derive(Debug)]
pub struct SecureSubmissionResult {
    pub accepted: bool,
    pub audit: Vec<String>,
    pub score: f64,
}

/// High-level helper for submitting self-evolution proposals securely.
pub fn submit_self_evolution_proposal_securely(
    conductor: &LatticeConductorV14,
    proposal: &mut SelfEvolutionProposal,
    signer_id: &str,
    threshold: f64,
) -> SecureSubmissionResult {
    proposal.sign_with_post_quantum(signer_id);
    let (accepted, audit, score) = conductor.submit_secure_governance_proposal(proposal, threshold);

    SecureSubmissionResult {
        accepted,
        audit,
        score,
    }
}