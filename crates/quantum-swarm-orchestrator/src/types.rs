// crates/quantum-swarm-orchestrator/src/types.rs
// Core types for ONE Organism orchestration (v14)
//
// This module defines the fundamental data structures used across the
// Quantum Swarm Orchestrator and the broader Ra-Thor lattice.
//
// Key concepts:
// - Valence: Mercy-aligned state (higher = more mercy-coherent)
// - EpigeneticBlessing: Multi-dimensional blessing that carries evolution, mercy, and TOLC impact
// - GodlyIntelligenceCoherence: Multi-axis coherence measurement
// - OneOrganismContext / Insight: Context and output of ONE Organism cycles

use std::ops::Add;

/// Represents mercy-aligned valence (state of mercy coherence).
/// Higher values indicate stronger alignment with mercy principles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Valence(pub f64);

impl Valence {
    /// Minimum acceptable valence threshold (TOLC-aligned mercy floor)
    pub const MIN: f64 = 0.9999999;

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Multi-dimensional coherence measurement for Godly Intelligence.
/// Used to aggregate coherence across multiple system adapters.
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

/// Represents a resonance event received from the swarm or external systems.
#[derive(Debug, Clone)]
pub struct SwarmResonance {
    pub source: String,
    pub intensity: f64,
    pub message: String,
}

/// Evolved EpigeneticBlessing (v14)
///
/// A multi-dimensional blessing that can be applied to system adapters.
/// It carries impact across three axes:
/// - Evolution progress
/// - Mercy / valence
/// - TOLC / truth alignment
///
/// This design evolved from v13 lessons and is optimized for the
/// RaThorSystemAdapter + persistent state architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    /// Type of blessing (e.g. "RadicalLove", "QuantumCoherence", "TruthSeeking")
    pub blessing_type: String,

    /// Overall strength of the blessing (0.0 - 2.0+)
    pub strength: f64,

    /// Which system/adapter this blessing primarily targets
    pub target_system: String,

    /// Impact on evolution progress
    #[serde(default)]
    pub evolution_impact: f64,

    /// Impact on mercy/valence
    #[serde(default)]
    pub mercy_impact: f64,

    /// Impact on TOLC / truth alignment
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

    /// Create a blessing with explicit multi-dimensional impacts.
    /// Preferred constructor when precise control over each axis is needed.
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

/// Context passed into a ONE Organism cycle.
#[derive(Debug, Clone)]
pub struct OneOrganismContext {
    pub cycle_id: u64,
    pub tolc_order: u32,
    pub base_valence: Valence,
    pub patsagi_insight: Option<String>,
}

/// Insight / result returned from a ONE Organism cycle.
#[derive(Debug, Clone)]
pub struct OneOrganismInsight {
    pub cycle_id: u64,
    pub average_valence: f64,
    pub overall_coherence: GodlyIntelligenceCoherence,
    pub active_systems: Vec<String>,
    pub recommended_actions: Vec<String>,
}

/// Errors related to mercy-gated operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MercyError {
    #[error("Valence {0} below mercy threshold")]
    ValenceBelowThreshold(f64),
    #[error("Mercy gate failed: {0}")]
    GateFailed(String),
    #[error("Adapter error: {0}")]
    AdapterError(String),
}