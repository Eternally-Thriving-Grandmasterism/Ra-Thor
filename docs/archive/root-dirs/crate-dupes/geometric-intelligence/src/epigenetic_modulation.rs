//! EpigeneticModulation v14.5
//!
//! Core epigenetic feedback system for Ra-Thor ONE Organism.
//! Provides layer-aware modulation, evolution rate bonuses, and volatility control.
//! Fully integrated with PolyhedralHarmonicEngine layer transitions.
//! TOLC 8 Mercy Lattice aligned. PATSAGi Council reviewed. AG-SML v1.0.

use crate::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;

#[derive(Debug, Clone)]
pub struct EpigeneticModulation {
    pub strength: f64,
    pub volatility: f64,
    pub layer: String,
}

impl Default for EpigeneticModulation {
    fn default() -> Self {
        Self {
            strength: 0.75,
            volatility: 0.25,
            layer: "Platonic".to_string(),
        }
    }
}

impl EpigeneticModulation {
    pub fn new(strength: f64, volatility: f64, layer: &str) -> Self {
        Self {
            strength: strength.clamp(0.0, 2.0),
            volatility: volatility.clamp(0.0, 1.5),
            layer: layer.to_string(),
        }
    }

    /// Layer-specific epigenetic influence multiplier.
    /// Higher layers permit richer but still mercy-bounded modulation.
    pub fn layer_modulated_epigenetic_influence(&self) -> f64 {
        match self.layer.as_str() {
            "Platonic" => 0.95,
            "Archimedean" => 1.05,
            "Catalan" => 1.12,
            "Kepler-Poinsot" => 1.22,
            "U57-UniformStar" => 1.35,
            "Hyperbolic Tiling" => 1.48,
            _ => 1.0,
        }
    }

    /// Evolution rate bonus for Powrush RBE, particles, and future Real Estate hooks.
    /// Combines self strength, geometric harmony, and layer factor.
    pub fn evolution_rate_bonus(&self, geometric_harmony: f64) -> f64 {
        let layer_factor = self.layer_modulated_epigenetic_influence();
        let raw = (self.strength * 0.55 + geometric_harmony * 0.45) * layer_factor;
        raw.clamp(0.85, 2.2)
    }

    /// Volatility surge multiplier with safe bounds.
    pub fn volatility_surge_multiplier(&self) -> f64 {
        (1.0 + self.volatility * 0.6).clamp(1.0, 2.0)
    }

    /// Bidirectional feedback signal used by layer transition logic.
    pub fn compute_feedback_with_geometric_harmony(&self, geometric_harmony: f64) -> f64 {
        (self.strength * 0.5 + geometric_harmony * 0.5) * self.layer_modulated_epigenetic_influence()
    }

    /// Safely updates layer after a successful transition.
    /// Applies transition blessing strength boost.
    pub fn update_from_layer_transition(&mut self, new_layer: &str, transition_boost: f64) {
        if PolyhedralHarmonicEngine::LAYER_SEQUENCE.contains(&new_layer) {
            self.layer = new_layer.to_string();
            self.strength = (self.strength * transition_boost).clamp(0.5, 2.5);
            self.volatility = (self.volatility * 0.92 + 0.08).clamp(0.1, 1.3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_modulated_influence() {
        let mod_platonic = EpigeneticModulation::new(0.8, 0.2, "Platonic");
        let mod_hyper = EpigeneticModulation::new(0.8, 0.2, "Hyperbolic Tiling");
        assert!(mod_hyper.layer_modulated_epigenetic_influence() > mod_platonic.layer_modulated_epigenetic_influence());
    }

    #[test]
    fn test_evolution_rate_bonus() {
        let em = EpigeneticModulation::new(0.9, 0.3, "Catalan");
        let bonus = em.evolution_rate_bonus(0.88);
        assert!(bonus > 1.0 && bonus < 2.0);
    }

    #[test]
    fn test_update_from_layer_transition() {
        let mut em = EpigeneticModulation::default();
        em.update_from_layer_transition("Archimedean", 1.15);
        assert_eq!(em.layer, "Archimedean");
        assert!(em.strength > 0.75);
    }
}
