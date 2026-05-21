/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use std::collections::HashMap;
use std::fs;
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
    #[error("Persistence error: {0}")]
    Persistence(String),
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
        Self { name: name.to_string(), description: description.to_string(), potential_harm: potential_harm.clamp(0.0, 1.0) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MercyGate {
    HarmThreshold,
    KeywordFilter,
    TolcAlignment,
    ValenceProtection,
    PositiveEmotion,
    CouncilConsensus,
    SelfEvolutionAlignment,
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
            MercyGate::TolcAlignment => state.tolc_alignment >= 0.75 || operation.potential_harm <= 0.4,
            MercyGate::ValenceProtection => state.valence > 0.3,
            MercyGate::PositiveEmotion => true, // Placeholder
            MercyGate::CouncilConsensus => true, // Placeholder for PATSAGi
            MercyGate::SelfEvolutionAlignment => true, // Placeholder
            MercyGate::Custom(_) => true,
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

/// Metrics / Telemetry for the conductor
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConductorMetrics {
    pub total_ticks: u64,
    pub operations_processed: u64,
    pub mercy_violations: u64,
    pub councils_registered: u64,
}

#[derive(Debug)]
pub struct SimpleLatticeConductor {
    pub state: GeometricState,
    pub motor: BasicGeometricMotor,
    pub tick_count: u64,
    pub operation_history: Vec<Operation>,
    pub pending_operations: Vec<Operation>,
    pub mercy_violations: Vec<String>,
    pub councils: HashMap<u64, String>,
    pub metrics: ConductorMetrics,
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
            metrics: ConductorMetrics::default(),
        }
    }

    pub fn register_council(&mut self, id: u64, name: &str) {
        self.councils.insert(id, name.to_string());
        self.metrics.councils_registered += 1;
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
            MercyGate::PositiveEmotion,
        ];
        gates.iter().all(|gate| gate.check(operation, &self.state))
    }

    /// Persistence helpers
    pub fn save_to_file(&self, path: &str) -> ConductorResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ConductorError::Persistence(e.to_string()))?;
        fs::write(path, json).map_err(|e| ConductorError::Persistence(e.to_string()))?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> ConductorResult<Self> {
        let data = fs::read_to_string(path).map_err(|e| ConductorError::Persistence(e.to_string()))?;
        let conductor: Self = serde_json::from_str(&data)
            .map_err(|e| ConductorError::Persistence(e.to_string()))?;
        Ok(conductor)
    }
}

impl LatticeConductor for SimpleLatticeConductor {
    fn tick(&mut self) -> ConductorResult<()> {
        self.tick_count += 1;
        self.metrics.total_ticks += 1;

        // Process pending operations with mercy evaluation
        let mut remaining = Vec::new();
        for op in self.pending_operations.drain(..) {
            if self.evaluate_all_gates(&op) {
                self.operation_history.push(op);
                self.metrics.operations_processed += 1;
                self.state.mercy_score = (self.state.mercy_score + 0.02).min(1.0);
                self.state.valence = (self.state.valence + 0.01).min(1.0);
            } else {
                let violation = format!("Mercy violation: '{}'", op.name);
                self.mercy_violations.push(violation.clone());
                self.metrics.mercy_violations += 1;
                println!("[LatticeConductor] {}", violation);
                remaining.push(op);
            }
        }
        self.pending_operations = remaining;

        // More meaningful GeometricMotor integration
        // Simulate applying identity motor and slightly improving geometric coherence
        let _ = self.motor.apply_dual_quaternion(nalgebra::DualQuaternion::identity());
        self.state.tolc_alignment = (self.state.tolc_alignment + 0.005).min(1.0);

        // Natural drift
        self.state.mercy_score = (self.state.mercy_score + 0.005).min(1.0);
        self.state.valence = (self.state.valence + 0.005).min(1.0);

        Ok(())
    }

    fn conduct_council(&self, council_id: u64) -> ConductorResult<()> {
        if self.councils.contains_key(&council_id) {
            Ok(())
        } else {
            Err(ConductorError::Council(format!("Unknown council ID: {}", council_id)))
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
            self.metrics.mercy_violations += 1;
            println!("[Mercy] {}", violation);
        }
        self.operation_history.push(operation);
        self.metrics.operations_processed += 1;
        passes
    }

    pub fn get_mercy_violations(&self) -> &[String] {
        &self.mercy_violations
    }

    /// Basic connection point to PATSAGi councils
    pub fn get_registered_patsagi_councils(&self) -> Vec<u64> {
        self.councils.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_updates_geometric_state() {
        let mut conductor = SimpleLatticeConductor::new();
        let initial_tolc = conductor.state.tolc_alignment;
        let _ = conductor.tick();
        assert!(conductor.state.tolc_alignment >= initial_tolc);
    }

    #[test]
    fn persistence_roundtrip() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.register_council(1, "Test Council");
        let path = "/tmp/test_conductor.json";
        conductor.save_to_file(path).unwrap();
        let loaded = SimpleLatticeConductor::load_from_file(path).unwrap();
        assert_eq!(loaded.councils.len(), 1);
    }
}

pub mod prelude {
    pub use crate::{ConductorMetrics, ConductorResult, GeometricMotor, GeometricState, LatticeConductor, MercyGate, Operation, SimpleLatticeConductor};
    pub use crate::geometric::BasicGeometricMotor;
}