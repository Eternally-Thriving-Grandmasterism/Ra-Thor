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

/// Evolved EpigeneticBlessing (v14)
///
/// Incorporates lessons from lattice-conductor-v13 while being designed
/// for the RaThorSystemAdapter + persistent state architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    /// Type of blessing (e.g. "RadicalLove", "QuantumCoherence", "TruthSeeking")
    pub blessing_type: String,

    /// Overall strength of the blessing (0.0 - 2.0+)
    pub strength: f64,

    /// Which system/adapter this blessing primarily targets
    pub target_system: String,

    /// Impact on evolution progress (inspired by v13 evolution_boost)
    #[serde(default)]
    pub evolution_impact: f64,

    /// Impact on mercy/valence (inspired by v13 mercy_boost)
    #[serde(default)]
    pub mercy_impact: f64,

    /// Impact on TOLC / truth alignment (inspired by v13 tolc_boost)
    #[serde(default)]
    pub tolc_impact: f64,
}

impl EpigeneticBlessing {
    pub fn new(
        blessing_type: &str,
        strength: f64,
        target_system: &str,
    ) -> Self {
        Self {
            blessing_type: blessing_type.to_string(),
            strength,
            target_system: target_system.to_string(),
            evolution_impact: strength * 0.6,
            mercy_impact: strength * 0.3,
            tolc_impact: 0.02,
        }
    }

    /// Create a blessing with explicit multi-dimensional impacts (v13 style)
    pub fn with_impacts(
        blessing_type: &str,
        strength: f64,
        target_system: &str,
        evolution_impact: f64,
        mercy_impact: f64,
        tolc_impact: f64,
    ) -> Self {
        Self {
            blessing_type: blessing_type.to_string(),
            strength,
            target_system: target_system.to_string(),
            evolution_impact,
            mercy_impact,
            tolc_impact,
        }
    }
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