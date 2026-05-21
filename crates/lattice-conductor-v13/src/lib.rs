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

#[derive(Debug, Clone)]
pub struct Operation {
    pub name: String,
    pub description: String,
    pub potential_harm: f64,
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
    pub operation_history: Vec<Operation>,
    pub mercy_violations: Vec<String>,
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            state: GeometricState::new(),
            motor: BasicGeometricMotor::new(),
            tick_count: 0,
            operation_history: Vec::new(),
            mercy_violations: Vec::new(),
        }
    }

    fn check_mercy_gates(&self, operation: &Operation) -> bool {
        if operation.potential_harm > 0.7 {
            return false;
        }

        let harmful_keywords = ["harm", "destroy", "exploit", "manipulate", "deceive"];
        let name_lower = operation.name.to_lowercase();
        for keyword in harmful_keywords {
            if name_lower.contains(keyword) {
                return false;
            }
        }

        if self.state.mercy_score < 0.6 && operation.potential_harm > 0.3 {
            return false;
        }

        true
    }

    /// More sophisticated mercy rules including TOLC alignment impact
    fn evaluate_mercy_impact(&self, operation: &Operation) -> (bool, f64) {
        let base_pass = self.check_mercy_gates(operation);

        // TOLC alignment penalty
        let tolc_penalty = if self.state.tolc_alignment < 0.8 {
            operation.potential_harm * 0.5
        } else {
            0.0
        };

        let final_harm = operation.potential_harm + tolc_penalty;
        let passes = base_pass && final_harm <= 0.75;

        (passes, final_harm)
    }
}

impl LatticeConductor for SimpleLatticeConductor {
    fn tick(&mut self) -> ConductorResult<()> {
        self.tick_count += 1;

        // Simulate natural mercy_score and valence drift over time
        self.state.mercy_score = (self.state.mercy_score + 0.01).min(1.0);
        self.state.valence = (self.state.valence + 0.005).min(1.0);

        Ok(())
    }

    fn conduct_council(&self, _council_id: u64) -> ConductorResult<()> {
        Ok(())
    }

    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()> {
        Ok(())
    }

    fn validate_mercy(&self, operation: &Operation) -> bool {
        let (passes, _impact) = self.evaluate_mercy_impact(operation);
        passes
    }

    fn get_geometric_state(&self) -> GeometricState {
        self.state.clone()
    }
}

// Extension methods for operation tracking and mercy violation logging
impl SimpleLatticeConductor {
    pub fn register_operation(&mut self, operation: Operation) -> bool {
        let passes = self.validate_mercy(&operation);

        if !passes {
            let violation = format!("Mercy violation: '{}' (harm: {:.2})", operation.name, operation.potential_harm);
            self.mercy_violations.push(violation.clone());
            println!("[Mercy] {}", violation); // Simple logging
        }

        self.operation_history.push(operation);
        passes
    }

    pub fn get_mercy_violations(&self) -> &[String] {
        &self.mercy_violations
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
    }

    #[test]
    fn mercy_validation_rejects_high_harm() {
        let conductor = SimpleLatticeConductor::new();
        let op = Operation::new("Exploit", "Bad action", 0.9);
        assert!(!conductor.validate_mercy(&op));
    }

    #[test]
    fn register_operation_tracks_history_and_violations() {
        let mut conductor = SimpleLatticeConductor::new();
        let good = Operation::new("Help", "Support others", 0.1);
        let bad = Operation::new("Exploit Users", "Harmful", 0.85);

        assert!(conductor.register_operation(good));
        assert!(!conductor.register_operation(bad));

        assert_eq!(conductor.operation_history.len(), 2);
        assert_eq!(conductor.mercy_violations.len(), 1);
    }
}

pub mod prelude {
    pub use crate::{ConductorResult, GeometricMotor, GeometricState, LatticeConductor, Operation, SimpleLatticeConductor};
    pub use crate::geometric::BasicGeometricMotor;
}