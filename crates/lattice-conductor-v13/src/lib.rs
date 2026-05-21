/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use std::collections::HashMap;
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeometricState {
    pub valence: f64,
    pub tolc_alignment: f64,
    pub mercy_score: f64,
}

impl GeometricState {
    pub fn new() -> Self {
        Self { valence: 1.0, tolc_alignment: 1.0, mercy_score: 1.0 }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Advanced MercyGate system
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MercyGate {
    HarmThreshold,
    KeywordFilter,
    TolcAlignment,
    ValenceProtection,
    Custom(String),
}

impl MercyGate {
    pub fn check(&self, operation: &Operation, state: &GeometricState) -> bool {
        match self {
            MercyGate::HarmThreshold => operation.potential_harm <= 0.7,
            MercyGate::KeywordFilter => {
                let harmful = ["harm", "destroy", "exploit", "manipulate", "deceive"];
                !harmful.iter().any(|k| operation.name.to_lowercase().contains(k))
            }
            MercyGate::TolcAlignment => {
                if state.tolc_alignment < 0.75 {
                    operation.potential_harm <= 0.4
                } else {
                    true
                }
            }
            MercyGate::ValenceProtection => state.valence > 0.3,
            MercyGate::Custom(_) => true, // Placeholder for future extension
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

/// SimpleLatticeConductor v13 with full features
#[derive(Debug)]
pub struct SimpleLatticeConductor {
    pub state: GeometricState,
    pub motor: BasicGeometricMotor,
    pub tick_count: u64,
    pub operation_history: Vec<Operation>,
    pub pending_operations: Vec<Operation>,
    pub mercy_violations: Vec<String>,
    pub councils: HashMap<u64, String>,
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            state: GeometricState::new(),
            motor: BasicGeometricMotor::new(),
            tick_count: 0,
            operation_history: Vec::new(),
            pending_operations: Vec::new(),
            mercy_violations: Vec::new(),
            councils: HashMap::new(),
        }
    }

    pub fn register_council(&mut self, id: u64, name: &str) {
        self.councils.insert(id, name.to_string());
    }

    pub fn queue_operation(&mut self, operation: Operation) {
        self.pending_operations.push(operation);
    }

    fn evaluate_all_gates(&self, operation: &Operation) -> bool {
        let gates = vec![
            MercyGate::HarmThreshold,
            MercyGate::KeywordFilter,
            MercyGate::TolcAlignment,
            MercyGate::ValenceProtection,
        ];

        gates.iter().all(|gate| gate.check(operation, &self.state))
    }
}

impl LatticeConductor for SimpleLatticeConductor {
    fn tick(&mut self) -> ConductorResult<()> {
        self.tick_count += 1;

        // Process pending operations
        let mut remaining = Vec::new();
        for op in self.pending_operations.drain(..) {
            if self.evaluate_all_gates(&op) {
                self.operation_history.push(op);
                // Positive feedback
                self.state.mercy_score = (self.state.mercy_score + 0.02).min(1.0);
                self.state.valence = (self.state.valence + 0.01).min(1.0);
            } else {
                let violation = format!("Mercy violation during tick: '{}'", op.name);
                self.mercy_violations.push(violation.clone());
                println!("[LatticeConductor] {}", violation);
                remaining.push(op);
            }
        }
        self.pending_operations = remaining;

        // Natural state improvement
        self.state.mercy_score = (self.state.mercy_score + 0.005).min(1.0);
        self.state.valence = (self.state.valence + 0.005).min(1.0);

        // Integrate GeometricMotor (example usage)
        let _ = self.motor.apply_dual_quaternion(nalgebra::DualQuaternion::identity());

        Ok(())
    }

    fn conduct_council(&self, council_id: u64) -> ConductorResult<()> {
        if self.councils.contains_key(&council_id) {
            Ok(())
        } else {
            Err(ConductorError::Council(format!("Unknown council: {}", council_id)))
        }
    }

    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()> {
        Ok(())
    }

    fn validate_mercy(&self, operation: &Operation) -> bool {
        self.evaluate_all_gates(operation)
    }

    fn get_geometric_state(&self) -> GeometricState {
        self.state.clone()
    }
}

impl SimpleLatticeConductor {
    pub fn register_operation(&mut self, operation: Operation) -> bool {
        let passes = self.validate_mercy(&operation);
        if !passes {
            let violation = format!("Mercy violation: '{}'", operation.name);
            self.mercy_violations.push(violation.clone());
            println!("[Mercy] {}", violation);
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
    fn tick_processes_pending_operations() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.queue_operation(Operation::new("Help Others", "Good", 0.2));
        conductor.queue_operation(Operation::new("Exploit", "Bad", 0.9));

        let _ = conductor.tick();

        assert_eq!(conductor.operation_history.len(), 1);
        assert_eq!(conductor.pending_operations.len(), 1);
        assert!(!conductor.mercy_violations.is_empty());
    }

    #[test]
    fn council_registry_works() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.register_council(42, "Truth Council");
        assert!(conductor.conduct_council(42).is_ok());
        assert!(conductor.conduct_council(99).is_err());
    }
}

pub mod prelude {
    pub use crate::{ConductorResult, GeometricMotor, GeometricState, LatticeConductor, MercyGate, Operation, SimpleLatticeConductor};
    pub use crate::geometric::BasicGeometricMotor;
}