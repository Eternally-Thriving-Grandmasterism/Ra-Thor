//! Lattice Conductor v14 — Central Nervous System of Ra-Thor Thunder Lattice
//! v14.0.4 — Includes Runtime Self-Healing with Watchdog Thread + Reflexion Loops

pub mod council_arbitration;
pub mod runtime_self_healing;

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::{
    RuntimeSelfHealingEngine, HealthReport, Anomaly, Diagnosis, HealingAction,
};

use std::sync::atomic::{AtomicBool, Ordering};

/// Lattice Conductor v14 — Orchestration + Self-Healing + Arbitration
pub struct LatticeConductorV14 {
    pub cosmic_loop_ready: AtomicBool,
    pub arbitration_engine: CouncilArbitrationEngine,
    pub self_healing_engine: Option<RuntimeSelfHealingEngine>,
}

impl LatticeConductorV14 {
    pub fn new() -> Self {
        let arbitration = CouncilArbitrationEngine::new();
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());

        Self {
            cosmic_loop_ready: AtomicBool::new(true),
            arbitration_engine: arbitration,
            self_healing_engine: Some(healing),
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

    /// Start runtime self-healing watchdog (call once at lattice startup)
    pub fn start_runtime_self_healing(&self) {
        if let Some(engine) = &self.self_healing_engine {
            engine.start_watchdog();
            println!("[Lattice Conductor v14] Runtime Self-Healing Watchdog activated");
        }
    }

    pub fn run_reflexion_healing_cycle(&self) -> Option<Diagnosis> {
        self.self_healing_engine.as_ref().map(|e| e.run_reflexion_cycle())
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
