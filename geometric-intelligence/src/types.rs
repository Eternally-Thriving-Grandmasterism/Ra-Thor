//! Shared types for geometric intelligence layer

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
