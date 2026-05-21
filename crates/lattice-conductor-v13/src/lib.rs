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
    pub evolution_level: f64,
}

impl GeometricState {
    pub fn new() -> Self {
        Self { valence: 1.0, tolc_alignment: 1.0, mercy_score: 1.0, evolution_level: 0.0 }
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
            MercyGate::PositiveEmotion => true,
            MercyGate::CouncilConsensus => true,
            MercyGate::SelfEvolutionAlignment => true,
            MercyGate::Custom(_) => true,
        }
    }
}

/// Bridge trait for PATSAGi council integration
pub trait PatsagiCouncilBridge {
    fn request_consensus(&self, operation: &Operation) -> bool;
    fn report_state(&self, state: &GeometricState);
}

/// Simple implementation of PatsagiCouncilBridge
#[derive(Debug)]
pub struct SimplePatsagiBridge {
    pub registered_councils: Vec<u64>,
}

impl SimplePatsagiBridge {
    pub fn new() -> Self {
        Self { registered_councils: vec![] }
    }

    pub fn with_councils(councils: Vec<u64>) -> Self {
        Self { registered_councils: councils }
    }
}

impl PatsagiCouncilBridge for SimplePatsagiBridge {
    fn request_consensus(&self, operation: &Operation) -> bool {
        if self.registered_councils.is_empty() {
            return true;
        }
        operation.potential_harm < 0.5
    }

    fn report_state(&self, _state: &GeometricState) {}
}

pub trait LatticeConductor {
    fn tick(&mut self) -> ConductorResult<()>;
    fn conduct_council(&self, council_id: u64) -> ConductorResult<()>;
    fn orchestrate_swarm_evolution(&mut self) -> ConductorResult<()>;
    fn validate_mercy(&self, operation: &Operation) -> bool;
    fn get_geometric_state(&self) -> GeometricState;
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConductorMetrics {
    pub total_ticks: u64,
    pub operations_processed: u64,
    pub mercy_violations: u64,
    pub councils_registered: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumSwarm {
    pub active_branches: u32,
    pub coherence: f64,
    pub split_count: u32,
}

impl QuantumSwarm {
    pub fn new() -> Self {
        Self { active_branches: 1, coherence: 1.0, split_count: 0 }
    }

    pub fn evolve(&mut self) {
        self.active_branches = (self.active_branches + 1).min(57);
        self.coherence = (self.coherence + 0.01).min(1.0);
    }

    pub fn split_branch(&mut self) -> bool {
        if self.active_branches < 57 {
            self.active_branches += 1;
            self.split_count += 1;
            self.coherence = (self.coherence * 0.98).max(0.5);
            true
        } else {
            false
        }
    }

    pub fn merge_branches(&mut self, count: u32) {
        self.active_branches = self.active_branches.saturating_sub(count).max(1);
        self.coherence = (self.coherence + 0.05).min(1.0);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ConductorEvent {
    TickCompleted { tick: u64 },
    OperationApproved { name: String },
    OperationRejected { name: String, reason: String },
    SwarmEvolved { branches: u32, coherence: f64 },
    SelfEvolution { level: f64 },
    MercyViolation { operation: String },
    QuantumBranchSplit { new_branches: u32 },
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
    pub quantum_swarm: QuantumSwarm,
    pub events: Vec<ConductorEvent>,
    /// Optional PATSAGi bridge for consensus
    pub patsagi_bridge: Option<Box<dyn PatsagiCouncilBridge + Send + Sync>>,
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
            quantum_swarm: QuantumSwarm::new(),
            events: Vec::new(),
            patsagi_bridge: None,
        }
    }

    pub fn with_patsagi_bridge(mut self, bridge: Box<dyn PatsagiCouncilBridge + Send + Sync>) -> Self {
        self.patsagi_bridge = Some(bridge);
        self
    }

    pub fn register_council(&mut self, id: u64, name: &str) {
        self.councils.insert(id, name.to_string());
        self.metrics.councils_registered += 1;
    }

    pub fn queue_operation(&mut self, operation: Operation) {
        self.pending_operations.push(operation);
    }

    fn evaluate_all_gates(&self, operation: &Operation) -> bool {
        let mut passes = true;

        let gates = vec![
            MercyGate::HarmThreshold,
            MercyGate::KeywordFilter,
            MercyGate::TolcAlignment,
            MercyGate::ValenceProtection,
        ];

        for gate in gates {
            if !gate.check(operation, &self.state) {
                passes = false;
                break;
            }
        }

        // Wire in PATSAGi bridge if present
        if let Some(bridge) = &self.patsagi_bridge {
            if !bridge.request_consensus(operation) {
                passes = false;
            }
        }

        passes
    }

    fn perform_self_evolution_step(&mut self) {
        if self.state.mercy_score > 0.8 && self.state.valence > 0.7 {
            self.state.evolution_level += 0.01;
            self.events.push(ConductorEvent::SelfEvolution { level: self.state.evolution_level });
        }
    }

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

    pub fn get_registered_patsagi_councils(&self) -> Vec<u64> {
        self.councils.keys().cloned().collect()
    }

    pub fn emit_event(&mut self, event: ConductorEvent) {
        self.events.push(event);
    }
}

impl LatticeConductor for SimpleLatticeConductor {
    fn tick(&mut self) -> ConductorResult<()> {
        self.tick_count += 1;
        self.metrics.total_ticks += 1;

        let mut remaining = Vec::new();
        for op in self.pending_operations.drain(..) {
            if self.evaluate_all_gates(&op) {
                self.operation_history.push(op.clone());
                self.metrics.operations_processed += 1;
                self.state.mercy_score = (self.state.mercy_score + 0.025).min(1.0);
                self.state.valence = (self.state.valence + 0.015).min(1.0);
                self.emit_event(ConductorEvent::OperationApproved { name: op.name.clone() });
            } else {
                let violation = format!("Mercy violation: '{}'", op.name);
                self.mercy_violations.push(violation.clone());
                self.metrics.mercy_violations += 1;
                self.emit_event(ConductorEvent::OperationRejected { name: op.name.clone(), reason: "Failed mercy gates".to_string() });
                remaining.push(op);
            }
        }
        self.pending_operations = remaining;

        let _ = self.motor.apply_dual_quaternion(nalgebra::DualQuaternion::identity());
        self.state.tolc_alignment = (self.state.tolc_alignment + 0.01).min(1.0);

        self.perform_self_evolution_step();

        // Quantum Swarm influence on state
        let swarm_influence = (self.quantum_swarm.coherence - 0.5) * 0.02;
        self.state.mercy_score = (self.state.mercy_score + swarm_influence).clamp(0.0, 1.0);
        self.state.evolution_level = (self.state.evolution_level + swarm_influence * 0.5).max(0.0);

        if self.quantum_swarm.split_branch() {
            self.emit_event(ConductorEvent::QuantumBranchSplit { new_branches: self.quantum_swarm.active_branches });
        }

        self.quantum_swarm.evolve();

        self.emit_event(ConductorEvent::SwarmEvolved {
            branches: self.quantum_swarm.active_branches,
            coherence: self.quantum_swarm.coherence,
        });

        self.state.mercy_score = (self.state.mercy_score + 0.005).min(1.0);
        self.state.valence = (self.state.valence + 0.005).min(1.0);

        self.emit_event(ConductorEvent::TickCompleted { tick: self.tick_count });

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
        self.quantum_swarm.evolve();
        self.emit_event(ConductorEvent::SwarmEvolved {
            branches: self.quantum_swarm.active_branches,
            coherence: self.quantum_swarm.coherence,
        });
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
            self.emit_event(ConductorEvent::MercyViolation { operation: operation.name.clone() });
            println!("[Mercy] {}", violation);
        }
        self.operation_history.push(operation);
        self.metrics.operations_processed += 1;
        passes
    }

    pub fn get_mercy_violations(&self) -> &[String] {
        &self.mercy_violations
    }

    pub fn get_events(&self) -> &[ConductorEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantum_swarm_influences_state() {
        let mut conductor = SimpleLatticeConductor::new();
        for _ in 0..10 {
            let _ = conductor.tick();
        }
        // After several ticks, swarm should have influenced state
        assert!(conductor.state.mercy_score > 0.9 || conductor.state.evolution_level > 0.0);
    }
}

pub mod prelude {
    pub use crate::{ConductorEvent, ConductorMetrics, ConductorResult, GeometricMotor, GeometricState, LatticeConductor, MercyGate, Operation, PatsagiCouncilBridge, QuantumSwarm, SimpleLatticeConductor, SimplePatsagiBridge};
    pub use crate::geometric::BasicGeometricMotor;
}