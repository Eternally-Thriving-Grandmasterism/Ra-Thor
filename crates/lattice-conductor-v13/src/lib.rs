/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use thiserror::Error;

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

#[derive(Debug, Clone)]
pub struct GeometricState {
    pub valence: f64,
    pub tolc_alignment: f64,
    pub mercy_score: f64,
}

impl GeometricState {
    pub fn new() -> Self {
        Self {
            valence: 1.0,
            tolc_alignment: 1.0,
            mercy_score: 1.0,
        }
    }
}

/// Represents an operation that can be validated by mercy gates.
#[derive(Debug, Clone)]
pub struct Operation {
    pub name: String,
    pub description: String,
    pub potential_harm: f64, // 0.0 = no harm, 1.0 = severe harm
}

impl Operation {
    pub fn new(name: &str, description: &str, potential_harm: f64) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            potential_harm: potential_harm.clamp(0.0, 1.0),
        }
    }
}

pub trait LatticeConductor {
    fn tick(&mut self) -> ConductorResult<()>;
    fn conduct_council(&self, council_id: u64) -> ConductorResult<()>;
    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()>;
    fn validate_mercy(&self, operation: &Operation) -> bool;
    fn get_geometric_state(&self) -> GeometricState;
}

/// Simple in-memory implementation of LatticeConductor (v13)
#[derive(Debug)]
pub struct SimpleLatticeConductor {
    pub state: GeometricState,
    pub motor: BasicGeometricMotor,
    pub tick_count: u64,
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            state: GeometricState::new(),
            motor: BasicGeometricMotor::new(),
            tick_count: 0,
        }
    }

    /// Core mercy validation logic
    fn check_mercy_gates(&self, operation: &Operation) -> bool {
        // Rule 1: Reject operations with high potential harm
        if operation.potential_harm > 0.7 {
            return false;
        }

        // Rule 2: Operations containing harmful keywords are rejected
        let harmful_keywords = ["harm", "destroy", "exploit", "manipulate", "deceive"];
        let name_lower = operation.name.to_lowercase();
        for keyword in harmful_keywords {
            if name_lower.contains(keyword) {
                return false;
            }
        }

        // Rule 3: If current mercy_score is low, be more strict
        if self.state.mercy_score < 0.6 && operation.potential_harm > 0.3 {
            return false;
        }

        true
    }
}

impl LatticeConductor for SimpleLatticeConductor {
    fn tick(&mut self) -> ConductorResult<()> {
        self.tick_count += 1;
        Ok(())
    }

    fn conduct_council(&self, _council_id: u64) -> ConductorResult<()> {
        Ok(())
    }

    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()> {
        Ok(())
    }

    fn validate_mercy(&self, operation: &Operation) -> bool {
        self.check_mercy_gates(operation)
    }

    fn get_geometric_state(&self) -> GeometricState {
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_conductor_starts_with_valid_state() {
        let conductor = SimpleLatticeConductor::new();
        let state = conductor.get_geometric_state();
        assert_eq!(state.valence, 1.0);
        assert_eq!(state.tolc_alignment, 1.0);
    }

    #[test]
    fn simple_conductor_tick_increments_counter() {
        let mut conductor = SimpleLatticeConductor::new();
        let initial = conductor.tick_count;
        let _ = conductor.tick();
        assert_eq!(conductor.tick_count, initial + 1);
    }

    #[test]
    fn mercy_validation_rejects_high_harm_operations() {
        let conductor = SimpleLatticeConductor::new();
        let harmful_op = Operation::new("Exploit Users", "Exploit for profit", 0.85);
        assert!(!conductor.validate_mercy(&harmful_op));
    }

    #[test]
    fn mercy_validation_accepts_benign_operations() {
        let conductor = SimpleLatticeConductor::new();
        let good_op = Operation::new("Help Community", "Provide support", 0.1);
        assert!(conductor.validate_mercy(&good_op));
    }
}

pub mod prelude {
    pub use crate::{ConductorResult, GeometricMotor, GeometricState, LatticeConductor, Operation, SimpleLatticeConductor};
    pub use crate::geometric::BasicGeometricMotor;
}