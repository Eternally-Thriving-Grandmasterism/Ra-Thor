//! Lattice Conductor v14 — Central Nervous System of Ra-Thor Thunder Lattice
//! v14.0.6+ — Includes Distributed Mercy Mesh + Phase 14.1 Governance Extraction

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod governance;  // NEW Phase 14.1

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::{
    RuntimeSelfHealingEngine, HealthReport, Anomaly, Diagnosis, HealingAction,
};
pub use distributed_mercy_mesh::{DistributedMercyMesh, MercyEvent, MercyMeshConfig};
pub use governance::{
    mercy_weighted_quadratic_voting::{
        MercyWeightedVote, tally_mercy_weighted_quadratic_votes, proposal_passes_mercy_quadratic,
    },
    enhanced_exponential_conviction_staking::{
        ConvictionStake, apply_enhanced_exponential_conviction, score_self_evolution_proposal_with_mercy,
    },
};

use std::sync::atomic::{AtomicBool, Ordering};

/// Lattice Conductor v14 — Orchestration + Self-Healing + Distributed Mercy + Governance
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

    pub fn trigger_mercy_mesh_healing(&self, severity: f64) {
        if let Some(mesh) = &self.mercy_mesh {
            mesh.propagate_mercy_event(MercyEvent::HealingTriggered { severity });
            println!("[Lattice Conductor v14] Mercy Mesh healing event propagated");
        }
    }

    /// **Phase 14.1 — Highly visible and auditable mercy-gated governance cycle**
    /// This function orchestrates the full governance flow with explicit mercy alignment,
    /// quadratic voting, and conviction staking. Full audit trail produced.
    pub fn orchestrate_mercy_gated_governance_cycle(
        &self,
        proposal_id: &str,
        votes: &[MercyWeightedVote],
        stakes: &[ConvictionStake],
        threshold: f64,
    ) -> (bool, Vec<String>, f64) {
        println!("[LATTICE CONDUCTOR v14] Starting mercy-gated governance cycle for: {}", proposal_id);

        let (passes, vote_audit) = proposal_passes_mercy_quadratic(votes, threshold);
        let (self_evo_score, stake_metadata) = score_self_evolution_proposal_with_mercy(proposal_id, stakes);

        let mut full_audit = vote_audit;
        full_audit.extend(stake_metadata);
        full_audit.push(format!("[GOVERNANCE] Proposal {} | Passes: {} | Self-Evo Score: {:.2}", proposal_id, passes, self_evo_score));

        println!("[LATTICE CONDUCTOR v14] Governance cycle complete. Audit entries: {}", full_audit.len());

        (passes, full_audit, self_evo_score)
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