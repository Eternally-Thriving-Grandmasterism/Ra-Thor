/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use std::collections::{HashMap, VecDeque};
use std::fs;
use thiserror::Error;

pub type ConductorResult<T> = Result<T, ConductorError>;

/// Core geometric + mercy + evolution state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeometricState {
    pub valence: f64,
    pub tolc_alignment: f64,
    pub mercy_score: f64,
    pub evolution_level: f64,

    pub current_mercy_recovery_rate: f64,
    pub current_evolution_rate: f64,
}

impl GeometricState {
    pub fn new() -> Self {
        Self {
            valence: 1.0,
            tolc_alignment: 1.0,
            mercy_score: 1.0,
            evolution_level: 0.0,
            current_mercy_recovery_rate: 0.025,
            current_evolution_rate: 0.01,
        }
    }
}

/// Adaptive parameters with 4-layer adaptation (meta-meta-meta)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveParameters {
    // Layer 1: Core rates
    pub mercy_recovery_rate: f64,
    pub evolution_rate: f64,
    pub swarm_influence_strength: f64,

    // Layer 2: Meta speeds
    pub mercy_recovery_adaptation_speed: f64,
    pub evolution_rate_adaptation_speed: f64,

    // Layer 3: Meta-Meta factors (how meta-speeds adapt)
    pub meta_speed_decay_factor: f64,
    pub meta_speed_increase_factor: f64,

    // Layer 4 (Meta-Meta-Meta): How Layer 3 factors themselves adapt
    pub meta_factor_adaptation_speed: f64,
}

impl AdaptiveParameters {
    pub fn new() -> Self {
        Self {
            // Layer 1
            mercy_recovery_rate: 0.025,
            evolution_rate: 0.01,
            swarm_influence_strength: 0.02,

            // Layer 2
            mercy_recovery_adaptation_speed: 0.0002,
            evolution_rate_adaptation_speed: 0.0001,

            // Layer 3
            meta_speed_decay_factor: 0.995,
            meta_speed_increase_factor: 1.008,

            // Layer 4
            meta_factor_adaptation_speed: 0.00005,
        }
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

/// Mercy gates with dynamic strictness
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
    pub fn check(&self, operation: &Operation, state: &GeometricState, swarm_coherence: f64) -> bool {
        let dynamic_threshold = 0.7 * swarm_coherence;
        match self {
            MercyGate::HarmThreshold => operation.potential_harm <= dynamic_threshold,
            MercyGate::KeywordFilter => {
                let harmful = ["harm", "destroy", "exploit", "manipulate", "deceive"];
                !harmful.iter().any(|k| operation.name.to_lowercase().contains(k))
            }
            MercyGate::TolcAlignment => {
                let adjusted = if swarm_coherence < 0.7 { 0.85 } else { 0.75 };
                state.tolc_alignment >= adjusted || operation.potential_harm <= 0.4
            }
            MercyGate::ValenceProtection => {
                let required_valence = if swarm_coherence < 0.6 { 0.4 } else { 0.3 };
                state.valence > required_valence
            }
            MercyGate::PositiveEmotion => true,
            MercyGate::CouncilConsensus => true,
            MercyGate::SelfEvolutionAlignment => true,
            MercyGate::Custom(_) => true,
        }
    }
}

pub trait PatsagiCouncilBridge {
    fn request_consensus(&self, operation: &Operation) -> bool;
    fn report_state(&self, state: &GeometricState);
}

#[derive(Debug, Clone)]
pub struct Council {
    pub id: u64,
    pub name: String,
    pub weight: f64,
    pub council_type: String,
}

#[derive(Debug)]
pub struct SimplePatsagiBridge {
    pub councils: Vec<Council>,
    pub required_consensus_ratio: f64,
}

impl SimplePatsagiBridge {
    pub fn new() -> Self {
        Self { councils: vec![], required_consensus_ratio: 0.6 }
    }

    pub fn with_councils(councils: Vec<Council>) -> Self {
        Self { councils, required_consensus_ratio: 0.6 }
    }
}

impl PatsagiCouncilBridge for SimplePatsagiBridge {
    fn request_consensus(&self, operation: &Operation) -> bool {
        if self.councils.is_empty() {
            return true;
        }
        let total_weight: f64 = self.councils.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 { return true; }

        let mut yes_weight = 0.0;
        for council in &self.councils {
            let vote = if operation.potential_harm < 0.4 { 1.0 }
            else if operation.potential_harm > 0.7 { 0.0 }
            else {
                match council.council_type.as_str() {
                    "Mercy" => 0.9,
                    "Truth" => 0.7,
                    "Evolution" => 0.6,
                    _ => 0.5,
                }
            };
            yes_weight += vote * council.weight;
        }
        (yes_weight / total_weight) >= self.required_consensus_ratio
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

/// Quantum Swarm
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
    SwarmCoherenceChanged { old_coherence: f64, new_coherence: f64 },
}

pub trait ConductorObserver {
    fn on_event(&self, event: &ConductorEvent);

    fn is_interested_in(&self, event: &ConductorEvent) -> bool {
        true
    }

    fn priority(&self) -> i32 {
        0
    }
}

const MAX_EVENT_HISTORY: usize = 256;

#[derive(Debug)]
pub struct SimpleLatticeConductor {
    pub state: GeometricState,
    pub adaptive_params: AdaptiveParameters,
    pub motor: BasicGeometricMotor,
    pub tick_count: u64,
    pub operation_history: Vec<Operation>,
    pub pending_operations: Vec<Operation>,
    pub mercy_violations: Vec<String>,
    pub councils: HashMap<u64, String>,
    pub metrics: ConductorMetrics,
    pub quantum_swarm: QuantumSwarm,
    pub events: VecDeque<ConductorEvent>,
    pub patsagi_bridge: Option<Box<dyn PatsagiCouncilBridge + Send + Sync>>,
    pub observers: Vec<Box<dyn ConductorObserver + Send + Sync>>,
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            state: GeometricState::new(),
            adaptive_params: AdaptiveParameters::new(),
            motor: BasicGeometricMotor::new(),
            tick_count: 0,
            operation_history: Vec::new(),
            pending_operations: Vec::new(),
            mercy_violations: Vec::new(),
            councils: HashMap::new(),
            metrics: ConductorMetrics::default(),
            quantum_swarm: QuantumSwarm::new(),
            events: VecDeque::with_capacity(MAX_EVENT_HISTORY),
            patsagi_bridge: None,
            observers: Vec::new(),
        }
    }

    pub fn with_patsagi_bridge(mut self, bridge: Box<dyn PatsagiCouncilBridge + Send + Sync>) -> Self {
        self.patsagi_bridge = Some(bridge);
        self
    }

    pub fn register_observer(&mut self, observer: Box<dyn ConductorObserver + Send + Sync>) {
        self.observers.push(observer);
        self.observers.sort_by_key(|o| std::cmp::Reverse(o.priority()));
    }

    pub fn unregister_observer(&mut self, index: usize) {
        if index < self.observers.len() {
            self.observers.remove(index);
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
        let mut passes = true;
        for gate in [MercyGate::HarmThreshold, MercyGate::KeywordFilter, MercyGate::TolcAlignment, MercyGate::ValenceProtection] {
            if !gate.check(operation, &self.state, self.quantum_swarm.coherence) {
                passes = false;
                break;
            }
        }
        if let Some(bridge) = &self.patsagi_bridge {
            if !bridge.request_consensus(operation) {
                passes = false;
            }
        }
        passes
    }

    /// Self-evolution with full 4-layer adaptation (including meta-meta-meta)
    fn perform_self_evolution_step(&mut self) {
        let p = &mut self.adaptive_params;

        let evolution_boost = (self.quantum_swarm.coherence - 0.5) * p.swarm_influence_strength;

        if self.state.mercy_score > 0.75 && self.state.valence > 0.65 {
            self.state.evolution_level += p.evolution_rate + evolution_boost.max(0.0);
            self.emit_event(ConductorEvent::SelfEvolution { level: self.state.evolution_level });
        }

        let coherence = self.quantum_swarm.coherence;
        let valence = self.state.valence;

        // === Layer 1 adaptation (using Layer 2) ===
        if coherence > 0.75 && valence > 0.7 {
            p.evolution_rate += p.evolution_rate_adaptation_speed;
            p.evolution_rate = p.evolution_rate.min(0.06);
        }

        if self.metrics.mercy_violations > 2 {
            p.mercy_recovery_rate += p.mercy_recovery_adaptation_speed;
            p.mercy_recovery_rate = p.mercy_recovery_rate.min(0.07);
        }

        // === Layer 3: Adaptive adjustment of meta-speeds (using Layer 4) ===
        if coherence > 0.8 && valence > 0.75 {
            // Healthy → decay meta-speeds
            p.evolution_rate_adaptation_speed *= p.meta_speed_decay_factor;
            p.mercy_recovery_adaptation_speed *= p.meta_speed_decay_factor;

            p.evolution_rate_adaptation_speed = p.evolution_rate_adaptation_speed.max(0.00002);
            p.mercy_recovery_adaptation_speed = p.mercy_recovery_adaptation_speed.max(0.00003);
        } else if self.metrics.mercy_violations > 4 || coherence < 0.6 {
            // Stressed → increase meta-speeds
            p.evolution_rate_adaptation_speed *= p.meta_speed_increase_factor;
            p.mercy_recovery_adaptation_speed *= p.meta_speed_increase_factor;

            p.evolution_rate_adaptation_speed = p.evolution_rate_adaptation_speed.min(0.001);
            p.mercy_recovery_adaptation_speed = p.mercy_recovery_adaptation_speed.min(0.0015);
        }

        // === Layer 4 (Meta-Meta-Meta): Slowly adapt the meta-factors themselves ===
        // This is the deepest layer currently implemented
        let meta_adapt = p.meta_factor_adaptation_speed;

        if coherence > 0.85 && valence > 0.8 {
            // Very healthy long-term → make decay stronger (more stabilization)
            p.meta_speed_decay_factor = (p.meta_speed_decay_factor - meta_adapt).max(0.98);
        } else if self.metrics.mercy_violations > 6 || coherence < 0.55 {
            // Chronic stress → make increase factor stronger (more aggressive response)
            p.meta_speed_increase_factor = (p.meta_speed_increase_factor + meta_adapt * 2.0).min(1.02);
        }

        // Gentle drift of swarm influence
        let target = 0.022;
        p.swarm_influence_strength = p.swarm_influence_strength * 0.996 + target * 0.004;

        // Update visible state
        self.state.current_mercy_recovery_rate = p.mercy_recovery_rate;
        self.state.current_evolution_rate = p.evolution_rate;
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
        if self.events.len() == MAX_EVENT_HISTORY {
            self.events.pop_front();
        }
        self.events.push_back(event.clone());

        for observer in &self.observers {
            if observer.is_interested_in(&event) {
                observer.on_event(&event);
            }
        }
    }

    pub fn replay_recent_events_to(&self, observer: &dyn ConductorObserver, count: usize) {
        let start = if self.events.len() > count { self.events.len() - count } else { 0 };
        for event in self.events.range(start..) {
            if observer.is_interested_in(event) {
                observer.on_event(event);
            }
        }
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

                self.state.mercy_score = (self.state.mercy_score + self.adaptive_params.mercy_recovery_rate).min(1.0);
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

        let swarm_influence = (self.quantum_swarm.coherence - 0.5) * self.adaptive_params.swarm_influence_strength;
        self.state.mercy_score = (self.state.mercy_score + swarm_influence).clamp(0.0, 1.0);
        self.state.evolution_level = (self.state.evolution_level + swarm_influence * 0.5).max(0.0);

        if self.quantum_swarm.split_branch() {
            self.emit_event(ConductorEvent::QuantumBranchSplit { new_branches: self.quantum_swarm.active_branches });
        }

        let old_coherence = self.quantum_swarm.coherence;
        self.quantum_swarm.evolve();
        let new_coherence = self.quantum_swarm.coherence;

        self.emit_event(ConductorEvent::SwarmEvolved { branches: self.quantum_swarm.active_branches, coherence: new_coherence });
        self.emit_event(ConductorEvent::SwarmCoherenceChanged { old_coherence, new_coherence });

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
        let old_coherence = self.quantum_swarm.coherence;
        self.quantum_swarm.evolve();
        let new_coherence = self.quantum_swarm.coherence;

        self.emit_event(ConductorEvent::SwarmEvolved { branches: self.quantum_swarm.active_branches, coherence: new_coherence });
        self.emit_event(ConductorEvent::SwarmCoherenceChanged { old_coherence, new_coherence });
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

    pub fn get_events(&self) -> Vec<&ConductorEvent> {
        self.events.iter().collect()
    }

    pub fn replay_recent_events_to(&self, observer: &dyn ConductorObserver, count: usize) {
        let start = if self.events.len() > count { self.events.len() - count } else { 0 };
        for event in self.events.range(start..) {
            if observer.is_interested_in(event) {
                observer.on_event(event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_gate_threshold_with_low_coherence() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.quantum_swarm.coherence = 0.55;
        let borderline = Operation::new("Borderline Action", "Medium harm", 0.68);
        assert!(!conductor.validate_mercy(&borderline));
    }

    #[test]
    fn weighted_patsagi_voting() {
        let bridge = SimplePatsagiBridge::with_councils(vec![
            Council { id: 1, name: "Mercy Council".to_string(), weight: 2.0, council_type: "Mercy".to_string() },
            Council { id: 2, name: "Truth Council".to_string(), weight: 1.0, council_type: "Truth".to_string() },
        ]);
        let medium_harm = Operation::new("Complex Decision", "Medium", 0.55);
        assert!(bridge.request_consensus(&medium_harm));
    }

    #[test]
    fn quantum_swarm_affects_evolution_speed() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.quantum_swarm.coherence = 0.9;
        for _ in 0..30 {
            let _ = conductor.tick();
        }
        assert!(conductor.state.evolution_level > 0.2);
    }
}

pub mod prelude {
    pub use crate::{AdaptiveParameters, ConductorEvent, ConductorMetrics, ConductorObserver, ConductorResult, GeometricMotor, GeometricState, LatticeConductor, MercyGate, Operation, PatsagiCouncilBridge, QuantumSwarm, SimpleLatticeConductor, SimplePatsagiBridge};
    pub use crate::geometric::BasicGeometricMotor;
}