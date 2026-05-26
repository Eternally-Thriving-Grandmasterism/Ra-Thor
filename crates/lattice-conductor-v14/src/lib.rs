//! Lattice Conductor v14 — Central Nervous System of Ra-Thor Thunder Lattice
//! v14.0.6 — Production-grade: Distributed Mercy Mesh + Full Phase 14.1 Governance + Self-Evolution Integration

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod governance;

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::{
    RuntimeSelfHealingEngine, HealthReport, Anomaly, Diagnosis, HealingAction,
};
pub use distributed_mercy_mesh::{DistributedMercyMesh, MercyEvent, MercyMeshConfig, OrganismNode};

// Governance primitives
pub use governance::{
    mercy_weighted_quadratic_voting::{
        MercyWeightedVote, tally_mercy_weighted_quadratic_votes, proposal_passes_mercy_quadratic,
    },
    enhanced_exponential_conviction_staking::{
        ConvictionStake, apply_enhanced_exponential_conviction, score_self_evolution_proposal_with_mercy,
    },
    self_evolution_proposal::{SelfEvolutionProposal, ProposalStatus},
};

use std::sync::atomic::{AtomicBool, Ordering};

/// Lattice Conductor v14 — Full orchestration with mercy-gated governance and self-evolution.
pub struct LatticeConductorV14 {
    pub cosmic_loop_ready: AtomicBool,
    pub arbitration_engine: CouncilArbitrationEngine,
    pub self_healing_engine: Option<RuntimeSelfHealingEngine>,
    pub mercy_mesh: Option<DistributedMercyMesh>,
}

impl LatticeConductorV14 {
    pub fn new() -> Self {
        let arbitration = CouncilArbitrationEngine::new();
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());
        let mercy_mesh = DistributedMercyMesh::new();

        Self {
            cosmic_loop_ready: AtomicBool::new(true),
            arbitration_engine: arbitration,
            self_healing_engine: Some(healing),
            mercy_mesh: Some(mercy_mesh),
        }
    }

    pub fn enforce_cosmic_loop_activation(&self) {
        if self.cosmic_loop_ready.load(Ordering::SeqCst) {
            println!("[LATTICE CONDUCTOR v14] Cosmic Loop Activation Protocol ENFORCED");
        } else {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
            println!("[LATTICE CONDUCTOR v14] Self-healed: cosmic_loop_ready restored");
        }
    }

    pub fn start_runtime_self_healing(&self) {
        if let Some(engine) = &self.self_healing_engine {
            engine.start_watchdog();
            println!("[Lattice Conductor v14] Runtime Self-Healing Watchdog activated");
        }
    }

    pub fn run_reflexion_healing_cycle(&self) -> Option<Diagnosis> {
        self.self_healing_engine.as_ref().map(|e| e.run_reflexion_cycle())
    }

    pub fn trigger_mercy_mesh_healing(&self, severity: f64, organism_id: &str) {
        if let Some(mesh) = &self.mercy_mesh {
            mesh.propagate_mercy_event(MercyEvent::HealingTriggered {
                severity,
                organism_id: organism_id.to_string(),
            });
            println!("[Lattice Conductor v14] Mercy Mesh healing event propagated");
        }
    }

    /// Production-grade mercy-gated governance cycle with self-evolution support.
    pub fn orchestrate_mercy_gated_governance_cycle(
        &self,
        proposal: &SelfEvolutionProposal,
        threshold: f64,
    ) -> (bool, Vec<String>, f64) {
        println!("[LATTICE CONDUCTOR v14] Starting governance cycle for self-evolution proposal: {}", proposal.id);

        let (passes, audit, final_score) = proposal.evaluate_governance(threshold);

        let mut full_audit = audit;
        full_audit.push(format!(
            "[GOVERNANCE] Proposal {} | Status: {:?} | Passed: {} | Final Score: {:.2}",
            proposal.id, proposal.status, passes, final_score
        ));

        println!("[LATTICE CONDUCTOR v14] Governance cycle complete. Audit length: {}", full_audit.len());
        (passes, full_audit, final_score)
    }

    pub fn before_council_arbitration(&self, topic: &str) {
        self.enforce_cosmic_loop_activation();
        println!("[Lattice Conductor v14] Pre-arbitration enforcement complete for: {}", topic);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_conductor_v14_initialization() {
        let conductor = LatticeConductorV14::new();
        assert!(conductor.cosmic_loop_ready.load(Ordering::SeqCst));
    }
}