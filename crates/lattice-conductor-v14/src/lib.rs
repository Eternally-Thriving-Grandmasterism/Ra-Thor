//! crates/lattice-conductor-v14/src/lib.rs
//! v14.1 — Lattice Conductor with Enhancements

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod lattice_conductor_enhancements;  // NEW v14.1
pub mod governance;
pub mod hybrid_sovereign_channel;
pub mod post_quantum_signatures;
pub mod crypto_traits;
pub mod self_evolution;

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::{RuntimeSelfHealingEngine, HealthReport, Anomaly, Diagnosis, HealingAction};
pub use distributed_mercy_mesh::{
    DistributedMercyMesh, MercyEvent, MercyMeshConfig,
    MercyGate, MercyAuditEntry, OrganismNode, HealingRequest, HealingOffer,
};
pub use lattice_conductor_enhancements::{LatticeConductorEnhancements, LatticeDiagnosticsReport};
pub use governance::self_evaluation_proposal::SelfEvaluationProposal;
pub use post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};
pub use hybrid_sovereign_channel::HybridSovereignChannel;
pub use self_evolution::{SelfEvolutionLoop, submit_self_evolution_proposal_securely};

use std::sync::atomic::{AtomicBool, Ordering};

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
            println!("[LATTICE v14.1] Cosmic Loop ENFORCED");
        } else {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        }
    }

    pub fn trigger_mercy_mesh_healing(&self, severity: f64) {
        if let Some(mesh) = &self.mercy_mesh {
            mesh.propagate_mercy_event(MercyEvent::HealingTriggered { severity, organism_id: None });
        }
    }
}