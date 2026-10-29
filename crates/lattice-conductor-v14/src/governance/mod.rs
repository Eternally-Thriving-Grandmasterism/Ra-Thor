//! Governance module (v14.8.2)
//! Submodules + cooperative game theory surface.
//!
//! Note: Previously both `governance.rs` and `governance/` existed, which is invalid in Rust.
//! This mod.rs is the single entry point.

pub mod self_evolution_proposal;
pub mod enhanced_exponential_conviction_staking;
pub mod mercy_weighted_quadratic_voting;

// Re-export the primary proposal type under both historical names for compatibility
pub use self_evolution_proposal::SelfEvolutionProposal;

// Alias for lib.rs historical export path
pub mod self_evaluation_proposal {
    pub use super::SelfEvolutionProposal as SelfEvaluationProposal;
}
