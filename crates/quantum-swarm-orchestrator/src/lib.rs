//! ra-thor-quantum-swarm-orchestrator
//! Quantum Swarm Orchestrator with ONE Organism Sovereign Health integration + Geometric Intelligence Layer
//!
//! Geometric intelligence is now delegated to the `geometric-intelligence` crate.

use std::sync::{Arc, RwLock};
use self_evolution::{SovereignHealthMonitor, init_sovereign_health_monitor};

pub mod quantum;
pub mod convergence;
pub mod integration;
pub mod tolc_seven_mercury_gates;

// Re-export geometric intelligence from the dedicated crate
pub use geometric_intelligence::*;

pub use convergence::*;
pub use integration::QuantumSwarmBridge;
pub use tolc_seven_mercury_gates::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Plasticity error: {0}")]
    Plasticity(String),
}

pub struct SwarmAgent {
    pub id: u64,
    pub mercury_valence: f64,
}

impl SwarmAgent {
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self { id: rng.gen(), mercury_valence: 0.55 + rng.gen_range(0.0..0.1) }
    }
    pub fn update_mercury_valence(&mut self, delta: f64) {
        self.mercury_valence = (self.mercury_valence + delta).clamp(0.0, 0.999);
    }
}

// ... (rest of the file remains, but now uses types from geometric_intelligence)
