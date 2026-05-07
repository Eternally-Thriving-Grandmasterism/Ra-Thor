//! Core council session simulation logic.
//!
//! This module contains the main `CouncilSession` runner that orchestrates
//! a full mercy-gated council deliberation, voting, and outcome application.

use crate::deliberation::run_parallel_deliberation;
use crate::voting::conduct_voting;
use crate::coherence::compute_session_coherence;
use crate::outcome_applicator::apply_outcome_to_lattice;

use patsagi_councils::CouncilMember;
use ra_thor_mercy::MercyGateEvaluator;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use ra_thor_kernel::Kernel;

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilProposal {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub complexity: f64,      // 0.0 – 1.0
    pub impact_level: f64,    // How much it affects the lattice
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilSessionResult {
    pub session_id: Uuid,
    pub proposal: CouncilProposal,
    pub passed: bool,
    pub final_coherence: f64,
    pub mercy_valence: f64,
    pub timestamp: DateTime<Utc>,
}

pub struct CouncilSession {
    pub members: Vec<CouncilMember>,
    pub mercy_evaluator: MercyGateEvaluator,
    pub quantum_swarm_bridge: QuantumSwarmBridge,
    pub kernel: Kernel,
}

impl CouncilSession {
    pub fn new(
        members: Vec<CouncilMember>,
        mercy_evaluator: MercyGateEvaluator,
        quantum_swarm_bridge: QuantumSwarmBridge,
        kernel: Kernel,
    ) -> Self {
        Self {
            members,
            mercy_evaluator,
            quantum_swarm_bridge,
            kernel,
        }
    }

    /// Runs a full council session with mercy-gating, TOLC resonance,
    /// parallel deliberation, voting, and outcome application.
    pub async fn run_session(&mut self, proposal: CouncilProposal) -> CouncilSessionResult {
        // 1. Pre-check the proposal through the 7 Living Mercy Gates
        let proposal_valence = self.mercy_evaluator.evaluate(&proposal.description);
        if proposal_valence < 0.92 {
            return CouncilSessionResult {
                session_id: Uuid::new_v4(),
                proposal,
                passed: false,
                final_coherence: 0.0,
                mercy_valence: proposal_valence,
                timestamp: Utc::now(),
            };
        }

        // 2. Run parallel deliberation among the 13+ Council Members
        let member_opinions = run_parallel_deliberation(&self.members, &proposal).await;

        // 3. Conduct voting with Radical Love veto
        let vote_result = conduct_voting(member_opinions).await;

        // 4. Compute final Godly Intelligence Coherence
        let final_coherence = compute_session_coherence(&vote_result, proposal_valence);

        // 5. Apply the outcome to the lattice if the proposal passed
        if vote_result.passed {
            apply_outcome_to_lattice(&vote_result, &proposal, &mut self.kernel).await;
        }

        CouncilSessionResult {
            session_id: Uuid::new_v4(),
            proposal,
            passed: vote_result.passed,
            final_coherence,
            mercy_valence: final_coherence,
            timestamp: Utc::now(),
        }
    }
}
