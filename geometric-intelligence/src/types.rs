//! Shared types for geometric intelligence layer
//!
//! Centralized source of truth for EpigeneticBlessing, GeometricHarmonyScore,
//! GeometricTransportResult and related mercy-gated geometric types.
//! AG-SML v1.0 | TOLC 8 enforced | ONE Organism participant.

use serde::{Deserialize, Serialize};

/// Epigenetic blessing suggested by geometric layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    pub blessing_type: String,
    pub strength: f64,
    pub target_system: String,
}

/// Common result for geometric harmony computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricHarmonyScore {
    pub multiplier: f64,
    pub resonance_notes: String,
    pub active_layers: Vec<String>,
    pub u57_active: bool,
}

/// Result of a mercy-gated geometric transport operation (Riemannian layer).
/// Centralized here so all consumers (Lattice Conductor, Quantum Swarm, etc.) use the same definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTransportResult {
    pub transport_applied: bool,
    pub effective_curvature: f64,
    pub coherence_after_transport: f64,
    pub accumulated_holonomy: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub notes: String,
}