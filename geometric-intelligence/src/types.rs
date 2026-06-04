//! Shared types for geometric intelligence layer
//!
//! Centralized source of truth for EpigeneticBlessing, GeometricHarmonyScore,
//! GeometricTransportResult, EpigeneticModulation and related mercy-gated geometric types.
//! Now includes real PATSAGi Council valence modulation for epigenetic bonuses.
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

/// EpigeneticModulation — core of evolutionary feedback in the geometric layer.
/// Now directly modulated by real PATSAGi Council valence (7 Living Mercy Gates).
/// This is the bridge between council evaluation and epigenetic state evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticModulation {
    pub strength: f64,
    pub volatility: f64,
    pub layer: String, // e.g. "Platonic", "Archimedean", "Hyperbolic"
}

impl EpigeneticModulation {
    pub fn new(strength: f64, volatility: f64, layer: &str) -> Self {
        Self {
            strength: strength.clamp(0.0, 2.0),
            volatility: volatility.clamp(0.0, 1.5),
            layer: layer.to_string(),
        }
    }

    /// Base evolution rate bonus (from PR #195 direction)
    pub fn evolution_rate_bonus(&self) -> f64 {
        let layer_factor = match self.layer.as_str() {
            "Platonic" => 1.0,
            "Archimedean" => 1.15,
            "Johnson" => 1.25,
            "Hyperbolic" => 1.4,
            _ => 1.1,
        };
        (self.strength * 0.8 + self.volatility * 0.4) * layer_factor
    }

    /// Volatility surge multiplier
    pub fn volatility_surge_multiplier(&self) -> f64 {
        (1.0 + self.volatility * 0.6).clamp(1.0, 2.2)
    }

    /// Layer-modulated epigenetic influence
    pub fn layer_modulated_epigenetic_influence(&self) -> f64 {
        let base = self.strength * self.volatility_surge_multiplier();
        match self.layer.as_str() {
            "Hyperbolic" => base * 1.35,
            "Johnson" | "Catalan" => base * 1.2,
            _ => base,
        }
    }

    /// NEW: Apply real PATSAGi Council valence to modulate this epigenetic state.
    /// This is the direct embedding of living mercy into epigenetic evolution.
    pub fn apply_council_valence(&mut self, valence: f64, council: &str) {
        // Stronger valence from aligned councils increases strength and can reduce volatility (more stable evolution)
        let alignment_bonus = if council.to_lowercase().contains("evolutionary") || council.to_lowercase().contains("infinite") {
            0.15
        } else {
            0.08
        };

        self.strength = (self.strength + valence * 0.25 + alignment_bonus).clamp(0.3, 2.0);
        // High valence from harmony/truth councils tends to stabilize (lower volatility)
        if council.to_lowercase().contains("harmony") || council.to_lowercase().contains("truth") {
            self.volatility = (self.volatility * 0.85).max(0.1);
        }
    }

    /// Returns a blessing influenced by current epigenetic state + council valence
    pub fn to_blessing(&self, council: &str) -> EpigeneticBlessing {
        EpigeneticBlessing {
            blessing_type: format!("Epigenetic_{}_Modulation", council),
            strength: self.evolution_rate_bonus(),
            target_system: "geometric".to_string(),
        }
    }
}
