/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use thiserror::Error;

/// Result type used throughout the Lattice Conductor.
pub type ConductorResult<T> = Result<T, ConductorError>;

#[derive(Debug, Error)]
pub enum ConductorError {
    #[error("Mercy violation: {0}")]
    MercyViolation(String),
    #[error("Geometric error: {0}")]
    Geometric(String),
    #[error("Council error: {0}")]
    Council(String),
}

/// Core state representing the geometric + mercy condition of the lattice.
#[derive(Debug, Clone)]
pub struct GeometricState {
    pub valence: f64,
    pub tolc_alignment: f64,
}

impl GeometricState {
    pub fn new() -> Self {
        Self {
            valence: 1.0,
            tolc_alignment: 1.0,
        }
    }

    pub fn valence(&self) -> f64 {
        self.valence
    }

    pub fn tolc_alignment(&self) -> f64 {
        self.tolc_alignment
    }
}

/// The central Lattice Conductor trait (v13).
pub trait LatticeConductor {
    /// Perform one conduction tick.
    fn tick(&mut self) -> ConductorResult<()>;

    /// Conduct a specific council.
    fn conduct_council(&self, council_id: u64) -> ConductorResult<()>;

    /// Orchestrate a round of swarm evolution.
    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()>;

    /// Validate that an operation passes mercy gates.
    fn validate_mercy(&self, operation: &str) -> bool;

    /// Return current geometric + mercy state.
    fn get_geometric_state(&self) -> GeometricState;
}

pub mod prelude {
    pub use crate::{ConductorResult, GeometricMotor, GeometricState, LatticeConductor};
    pub use crate::geometric::BasicGeometricMotor;
}