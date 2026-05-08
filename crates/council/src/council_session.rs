//! Core council session simulation logic with full Quantum Swarm Bridge integration.

use crate::deliberation::run_parallel_deliberation;
use crate::voting::conduct_voting;
use crate::coherence::compute_session_coherence;
use crate::outcome_applicator::apply_outcome_to_lattice;

use patsagi_councils::CouncilMember;
use ra_thor_mercy::MercyGateEvaluator;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;  // ← Full integration
use ra_thor_kernel::Kernel;
use powrush::PowrushGame;  // For spine cycle feedback

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilProposal {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub complexity: f64,
    pub impact_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilSessionResult {
    pub session_id: Uuid,
    pub proposal: CouncilProposal,
    pub passed: bool,
    pub final_coherence: f64,
    pub mercy_valence: f64,
    pub swarm_insight: Option<String>,          // ← New: Bridge output
    pub timestamp: DateTime<Utc>,
}

pub struct CouncilSession {
    pub members: Vec<CouncilMember>,
    pub mercy_evaluator: MercyGateEvaluator,
    pub quantum_swarm_bridge: QuantumSwarmBridge,   // ← Now properly injected
    pub kernel: Kernel,
    pub powrush_game: PowrushGame,                  // ← For spine cycle feedback
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
            powrush_game: PowrushGame::new(),  // Default game state
        }
    }

    pub async fn run_session(&mut self, proposal: CouncilProposal) -> CouncilSessionResult {
        // 1. Mercy pre-check
        let proposal_valence = self.mercy_evaluator.evaluate(&proposal.description);
        if proposal_valence < 0.92 {
            return self.blocked_result(proposal, proposal_valence);
        }

        // 2. Parallel deliberation
        let member_opinions = run_parallel_deliberation(&self.members, &proposal).await;

        // 3. Voting
        let vote_result = conduct_voting(member_opinions).await;

        // 4. Quantum Swarm Bridge integration (the heart of complex decisions)
        let swarm_insight = if proposal.complexity > 0.7 {
            let tolc_order = (proposal.impact_level * 33.0) as u32;  // Map impact to TOLC order
            Some(self.quantum_swarm_bridge
                .run_spine_coordinated_cycle(tolc_order, proposal_valence, &mut self.powrush_game)
                .await)
        } else {
            None
        };

        // 5. Coherence + outcome
        let final_coherence = compute_session_coherence(&vote_result, proposal_valence);

        if vote_result.passed {
            apply_outcome_to_lattice(&vote_result, &proposal, &mut self.kernel).await;
        }

        CouncilSessionResult {
            session_id: Uuid::new_v4(),
            proposal,
            passed: vote_result.passed,
            final_coherence,
            mercy_valence: proposal_valence,
            swarm_insight,
            timestamp: Utc::now(),
        }
    }

    fn blocked_result(&self, proposal: CouncilProposal, valence: f64) -> CouncilSessionResult {
        CouncilSessionResult {
            session_id: Uuid::new_v4(),
            proposal,
            passed: false,
            final_coherence: 0.0,
            mercy_valence: valence,
            swarm_insight: None,
            timestamp: Utc::now(),
        }
    }
}
