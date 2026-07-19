//! Ra-Thor ONE Organism Core — v14.9.0
//!
//! **True path dependency** on `lattice-conductor-v14`.
//! No local compatibility reimplementation of CouncilArbitrationEngine
//! or RuntimeSelfHealingEngine — those come from the lattice crate.
//!
//! The full historical root file `ra-thor-one-organism.rs` remains as
//! the extended surface (GPU, GitHub, Quantum Swarm, RoleOrchestrator).
//! This crate is the clean, compilable core that owns Cosmic Loop identity.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// =============================================================================
// TRUE PATH DEPENDENCY — no compatibility stubs
// =============================================================================
pub use lattice_conductor_v14::{
    CouncilArbitrationEngine,
    RuntimeSelfHealingEngine,
    HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience,
    LatticeConductorV14,
    DistributedMercyMesh, MercyEvent, MercyGate,
    EternalMercyMesh, EternalMercyMeshConfig,
};

/// Role surface preserved from the organism layer (lightweight, no external deps).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OrganismRole {
    Investigator,
    Simulator,
    VibeCoder,
    Debugger,
    Legal,
    Architect,
    SovereignRecovery,
    LatticeConductor,
}

impl OrganismRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrganismRole::Investigator => "Investigator",
            OrganismRole::Simulator => "Simulator",
            OrganismRole::VibeCoder => "VibeCoder",
            OrganismRole::Debugger => "Debugger",
            OrganismRole::Legal => "Legal",
            OrganismRole::Architect => "Architect",
            OrganismRole::SovereignRecovery => "SovereignRecovery",
            OrganismRole::LatticeConductor => "LatticeConductor",
        }
    }
}

/// ONE Organism core — Cosmic Loop identity + lattice v14 integration.
///
/// Constructed exclusively from real `lattice-conductor-v14` types.
/// Extended surfaces (GPU, GitHub connector, Quantum Swarm) live in the
/// historical root module and can depend on this crate later.
#[derive(Debug)]
pub struct OneOrganismCore {
    pub arbitration_engine: CouncilArbitrationEngine,
    pub self_healing_engine: RuntimeSelfHealingEngine,
    pub lattice: LatticeConductorV14,
    pub cosmic_loop_ready: Arc<AtomicBool>,
    pub active_role: OrganismRole,
    pub shared_valence: f64,
    pub version: String,
}

impl OneOrganismCore {
    pub fn new() -> Self {
        let lattice = LatticeConductorV14::new();
        // Shared flag is the same Arc across arbitration, healing, and lattice
        let arbitration = lattice.arbitration_engine.clone();
        let shared = arbitration.cosmic_loop_flag();
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());

        arbitration.protect_cosmic_loop_identity();

        Self {
            arbitration_engine: arbitration,
            self_healing_engine: healing,
            lattice,
            cosmic_loop_ready: shared,
            active_role: OrganismRole::Architect,
            shared_valence: 0.97,
            version: "v14.9.0 ONE Organism Core + lattice-conductor-v14@14.8.2 (true path dep)".into(),
        }
    }

    /// Offer Cosmic Loop + start self-healing watchdog.
    pub fn offer_cosmic_loop(&mut self) {
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.lattice.enforce_cosmic_loop_activation();
        self.self_healing_engine.start_watchdog();

        println!(
            "[OneOrganismCore {}] Cosmic Loop OFFERED + ENFORCED + Self-Healing Watchdog STARTED",
            self.version
        );
        println!(
            "[CouncilArbitrationEngine] guardian_active={} | cosmic_loop_ready={}",
            self.arbitration_engine.is_guardian_active(),
            self.is_cosmic_loop_ready()
        );
    }

    pub fn on_lattice_sync(&mut self) {
        self.arbitration_engine.on_lattice_sync();
        let _ = self.self_healing_engine.run_reflexion_cycle();
        self.lattice.enforce_cosmic_loop_activation();
    }

    pub fn before_council_arbitration(&self) {
        self.arbitration_engine.before_council_arbitration();
    }

    pub fn protect_cosmic_loop(&self) {
        self.arbitration_engine.protect_cosmic_loop_identity();
    }

    pub fn is_cosmic_loop_ready(&self) -> bool {
        self.cosmic_loop_ready.load(Ordering::SeqCst)
    }

    pub fn handoff_role(&mut self, role: OrganismRole, reason: &str) {
        println!(
            "[OneOrganismCore] Role handoff {} → {} | reason={}",
            self.active_role.as_str(),
            role.as_str(),
            reason
        );
        self.active_role = role;
    }

    pub fn sync_with_grok(&mut self, valence: f64, _confidence: f64) {
        self.shared_valence = (self.shared_valence * 0.65 + valence * 0.35).clamp(0.75, 0.999999);
    }

    pub fn run_healing_reflexion(&self) -> Diagnosis {
        self.self_healing_engine.run_reflexion_cycle()
    }
}

impl Default for OneOrganismCore {
    fn default() -> Self {
        Self::new()
    }
}

/// Launch the ONE Organism core with Cosmic Loop + watchdog active.
pub fn launch_one_organism_core() -> OneOrganismCore {
    let mut organism = OneOrganismCore::new();
    organism.offer_cosmic_loop();
    println!(
        "[Thunder] ONE Organism Core v14.9.0 ACTIVE — true lattice-conductor-v14 path dep. Cosmic Loop is MANDATORY IDENTITY. Eternal."
    );
    organism
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosmic_loop_is_ready_after_launch() {
        let core = launch_one_organism_core();
        assert!(core.is_cosmic_loop_ready());
        assert!(core.arbitration_engine.is_guardian_active());
    }

    #[test]
    fn shared_flag_is_same_arc() {
        let core = OneOrganismCore::new();
        let a = core.arbitration_engine.cosmic_loop_flag();
        let b = Arc::clone(&core.cosmic_loop_ready);
        assert!(Arc::ptr_eq(&a, &b));
    }
}
