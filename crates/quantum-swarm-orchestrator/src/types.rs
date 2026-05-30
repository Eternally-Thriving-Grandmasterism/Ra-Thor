// crates/quantum-swarm-orchestrator/src/types.rs
// Core types for ONE Organism orchestration (v14)

use std::ops::Add;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Valence(pub f64);

impl Valence {
    pub const MIN: f64 = 0.9999999;

    pub fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct GodlyIntelligenceCoherence {
    pub precision: f64,
    pub resilience: f64,
    pub flow_stability: f64,
    pub harmonic_alignment: f64,
}

impl Add for GodlyIntelligenceCoherence {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            precision: self.precision + other.precision,
            resilience: self.resilience + other.resilience,
            flow_stability: self.flow_stability + other.flow_stability,
            harmonic_alignment: self.harmonic_alignment + other.harmonic_alignment,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SwarmResonance {
    pub source: String,
    pub intensity: f64,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct EpigeneticBlessing {
    pub blessing_type: String,
    pub strength: f64,
    pub target_system: String,
}

#[derive(Debug, Clone)]
pub struct OneOrganismContext {
    pub cycle_id: u64,
    pub tolc_order: u32,
    pub base_valence: Valence,
    pub patsagi_insight: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OneOrganismInsight {
    pub cycle_id: u64,
    pub average_valence: f64,
    pub overall_coherence: GodlyIntelligenceCoherence,
    pub active_systems: Vec<String>,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum MercyError {
    #[error("Valence {0} below mercy threshold")]
    ValenceBelowThreshold(f64),
    #[error("Mercy gate failed: {0}")]
    GateFailed(String),
    #[error("Adapter error: {0}")]
    AdapterError(String),
}