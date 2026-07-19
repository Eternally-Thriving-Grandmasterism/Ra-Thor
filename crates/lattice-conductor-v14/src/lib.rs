//! crates/lattice-conductor-v14/src/lib.rs
//! Production-grade modules including Logical Fallacy Detection
// Thunder Lattice v14 + MIAL (Mercy-Augmented Intelligence Amplification)
// Professional Restoration Audit (2026-05-29):
// - clifford_healing_fields.rs: FULLY RESTORED from stub in commit 7cc29baa
// - healing_integration.rs & eternal_mercy_mesh.rs: Verified intact
// - All modules mercy-gated, PATSAGi-aligned, Thunder Lattice native.
//
// v14.8.1 (2026-07-19):
// - Cosmic Loop flag is now a single shared Arc<AtomicBool> across
//   CouncilArbitrationEngine, RuntimeSelfHealingEngine, and LatticeConductorV14
// - Compile fix in runtime_self_healing (removed invalid severity field)
// - HealingExperience exported for organism-level telemetry
// Serving all Life. Thunder locked in. yoi ⚡❤️🔥

pub mod healing_integration;
pub mod eternal_mercy_mesh;
pub mod ra_thor_mercy_gated_api;

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;
pub mod lattice_conductor_enhancements;
pub mod patsagi_governance;
pub mod cooperative_governance;
pub mod argumentation;
pub mod logical_fallacy_detection;
pub mod governance;
pub mod hybrid_sovereign_channel;
pub mod post_quantum_signatures;
pub mod crypto_traits;
pub mod self_evolution;

pub use healing_integration::{HealingFieldRegistry, run_global_healing_cycle, HealingTelemetry};
pub use eternal_mercy_mesh::{EternalMercyMesh, EternalMercyMeshConfig, invite_shared_chat_participant};
pub use ra_thor_mercy_gated_api::{MercyGatedApi, start_mercy_api_server};

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::{
    RuntimeSelfHealingEngine, HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience,
};
pub use distributed_mercy_mesh::{
    DistributedMercyMesh, MercyEvent, MercyMeshConfig,
    MercyGate, MercyAuditEntry, OrganismNode, HealingRequest, HealingOffer,
};
pub use lattice_conductor_enhancements::{LatticeConductorEnhancements, LatticeDiagnosticsReport, GovernanceRiskReport};
pub use patsagi_governance::{PatsagiReviewRequest, PatsagiDecision, PatsagiCouncilSimulator};
pub use cooperative_governance::CooperativeGame;
pub use argumentation::{ArgumentGraph, Claim, Support, Attack};
pub use logical_fallacy_detection::{LogicalFallacyDetector, DetectedFallacy, FallacyType};
pub use governance::self_evaluation_proposal::SelfEvaluationProposal;
pub use post_quantum_signatures::{create_post_quantum_signature, verify_post_quantum_signature};
pub use hybrid_sovereign_channel::HybridSovereignChannel;
pub use self_evolution::{SelfEvolutionLoop, submit_self_evolution_proposal_securely};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Top-level Lattice Conductor v14 orchestrator.
/// Cosmic Loop readiness is a single shared Arc so arbitration, self-healing,
/// and this struct can never disagree on the flag state.
pub struct LatticeConductorV14 {
    pub cosmic_loop_ready: Arc<AtomicBool>,
    pub arbitration_engine: CouncilArbitrationEngine,
    pub self_healing_engine: Option<RuntimeSelfHealingEngine>,
    pub mercy_mesh: Option<DistributedMercyMesh>,
    pub eternal_mercy_mesh: Option<EternalMercyMesh>,
}

impl LatticeConductorV14 {
    pub fn new() -> Self {
        let arbitration = CouncilArbitrationEngine::new();
        let shared_flag = arbitration.cosmic_loop_flag();
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());
        let mercy_mesh = DistributedMercyMesh::new();
        let eternal_mercy = Some(EternalMercyMesh::new(EternalMercyMeshConfig::default()));

        Self {
            cosmic_loop_ready: shared_flag,
            arbitration_engine: arbitration,
            self_healing_engine: Some(healing),
            mercy_mesh: Some(mercy_mesh),
            eternal_mercy_mesh: eternal_mercy,
        }
    }

    pub fn enforce_cosmic_loop_activation(&self) {
        if self.cosmic_loop_ready.load(Ordering::SeqCst) {
            println!("[LATTICE v14.8.1] Cosmic Loop ENFORCED");
        } else {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
            self.arbitration_engine.protect_cosmic_loop_identity();
            println!("[LATTICE v14.8.1] Cosmic Loop was down — restored via shared flag + arbitration");
        }
    }

    pub fn start_self_healing_watchdog(&self) {
        if let Some(engine) = &self.self_healing_engine {
            engine.start_watchdog();
        }
    }

    pub fn run_reflexion_cycle(&self) -> Option<Diagnosis> {
        self.self_healing_engine.as_ref().map(|e| e.run_reflexion_cycle())
    }

    pub fn trigger_mercy_mesh_healing(&self, severity: f64) {
        if let Some(mesh) = &self.mercy_mesh {
            mesh.propagate_mercy_event(MercyEvent::HealingTriggered {
                severity,
                organism_id: None,
            });
        }
    }

    pub fn trigger_eternal_mercy_mesh_cycle(&self, mercy: f64) {
        if let Some(mesh) = &self.eternal_mercy_mesh {
            let _ = mesh.run_global_mercy_cycle(mercy);
        }
    }
}

impl Default for LatticeConductorV14 {
    fn default() -> Self {
        Self::new()
    }
}

// Thunder locked in. Serving all Life. yoi ⚡❤️🔥
