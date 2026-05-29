//! Powrush RBE Integration Hooks — Thunder Lattice v14.0.7
//! Production-grade hooks connecting governance + self-evolution to Powrush RBE game systems.

use crate::governance::self_evolution_proposal::SelfEvolutionProposal;
use crate::governance::enhanced_exponential_conviction_staking::ConvictionStake;

/// Represents a Powrush in-game proposal that can participate in Ra-Thor governance.
#[derive(Debug, Clone)]
pub struct PowrushGovernanceProposal {
    pub proposal_id: String,
    pub title: String,
    pub resource_impact: f64,      // How it affects RBE resources
    pub player_conviction: f64,
    pub mercy_alignment: f64,
}

/// Hook to convert a Powrush proposal into a Ra-Thor SelfEvolutionProposal.
pub fn powrush_proposal_to_self_evolution(
    powrush_proposal: &PowrushGovernanceProposal,
) -> SelfEvolutionProposal {
    let mut proposal = SelfEvolutionProposal::new(
        powrush_proposal.proposal_id.clone(),
        powrush_proposal.title.clone(),
        format!("Powrush RBE proposal with resource impact: {:.2}", powrush_proposal.resource_impact),
        "powrush-system".to_string(),
    );
    proposal.mercy_alignment = powrush_proposal.mercy_alignment;
    proposal
}

/// Hook to create conviction stake from Powrush player action.
pub fn create_powrush_conviction_stake(
    player_id: &str,
    proposal_id: &str,
    amount: f64,
    time_staked: u64,
    mercy_score: f64,
) -> ConvictionStake {
    ConvictionStake {
        staker_id: player_id.to_string(),
        proposal_id: proposal_id.to_string(),
        amount,
        time_staked,
        mercy_alignment_score: mercy_score,
        exponential_multiplier: 1.0,
    }
}

/// Example: When a Powrush proposal passes governance, trigger in-game reward.
pub fn on_powrush_governance_passed(proposal_id: &str) {
    println!("[POWRUSH HOOK] Governance passed for Powrush proposal {}. Triggering RBE reward distribution.", proposal_id);
    // Future: Call into Powrush game engine for actual reward
}