//! TOLC Lattice Activation Engine
//!
//! The living heart of Ra-Thor's self-evolving mathematical lattice.
//! Version 0.5.25 — Significantly deeper mechanics with multi-order resonance,
//! stronger mercy amplification, and future-proof hooks for quantum swarm integration.

use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLCLatticeActivationEngine {
    pub current_max_order: u32,
    pub total_activations: u64,
    pub last_self_evolution_pulse: Option<String>,
    pub stability_score: f64,
    pub resonance_multiplier: f64,
}

impl TOLCLatticeActivationEngine {
    pub fn new() -> Self {
        Self {
            current_max_order: 80,
            total_activations: 0,
            last_self_evolution_pulse: None,
            stability_score: 0.999,
            resonance_multiplier: 1.0,
        }
    }

    /// Activate the lattice up to a target order (v0.5.25 — Deep Multi-Order Resonance)
    pub async fn activate_full_lattice_up_to(&mut self, target_order: u32, game: &mut PowrushGame) -> String {
        let effective_order = target_order.max(self.current_max_order);
        self.current_max_order = effective_order;
        self.total_activations += 1;

        // Deeper mercy-gated formula with multi-order resonance
        let mercy_valence = 0.975;
        let stability_factor = self.stability_score.min(0.9995);
        let resonance = self.resonance_multiplier;

        let activation_strength = 
            (effective_order as f64 * 0.0165 * resonance) 
            + (mercy_valence * 2.35 * stability_factor);

        let joy_boost = (activation_strength * 2850.0).min(215000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(9);

        // Increase resonance slightly with each activation
        self.resonance_multiplier = (self.resonance_multiplier + 0.008).min(1.35);
        self.stability_score = (self.stability_score + 0.0009).min(0.9996);

        let result = format!(
            "TOLC Lattice activated up to order {} (v0.5.25 Deep Multi-Order Resonance)\n\
             Activation Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             9-Gen Epigenetic Blessing applied\n\
             Resonance Multiplier: {:.3}\n\
             Stability Score: {:.4}\n\
             Total Activations: {}",
            effective_order,
            activation_strength,
            joy_boost,
            self.resonance_multiplier,
            self.stability_score,
            self.total_activations
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Stronger eternal self-evolution pulse (v0.5.25)
    pub fn quick_eternal_self_evolution_pulse(&mut self, game: &mut PowrushGame) -> String {
        let resonance = self.resonance_multiplier;
        let stability_factor = self.stability_score.min(0.9995);

        let pulse_strength = (self.current_max_order as f64 * 0.024 * resonance) + 16.0;
        let joy_boost = (pulse_strength * 3650.0 * stability_factor).min(255000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(11);

        self.stability_score = (self.stability_score + 0.0014).min(0.9997);
        self.resonance_multiplier = (self.resonance_multiplier + 0.012).min(1.42);

        let result = format!(
            "ETERNAL SELF-EVOLUTION PULSE (v0.5.25 Deep Mechanics)\n\
             Pulse Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             11-Gen Epigenetic Blessing applied\n\
             Resonance: {:.3} | Stability: {:.4}\n\
             Lattice Order: {}",
            pulse_strength,
            joy_boost,
            self.resonance_multiplier,
            self.stability_score,
            self.current_max_order
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Generate living cathedral status report (enhanced for v0.5.25)
    pub fn generate_living_cathedral_status_report(&self) -> String {
        format!(
            "=== TOLC Living Cathedral Status (v0.5.25) ===\n\
             Current Max Order: {}\n\
             Total Activations: {}\n\
             Stability Score: {:.4}\n\
             Resonance Multiplier: {:.3}\n\
             Last Self-Evolution Pulse: {}\n\
             Mercy-Gated: TRUE | Eternal Convergence: ACTIVE | Multi-Order Resonance: ENGAGED",
            self.current_max_order,
            self.total_activations,
            self.stability_score,
            self.resonance_multiplier,
            self.last_self_evolution_pulse.as_deref().unwrap_or("None yet")
        )
    }
}

impl Default for TOLCLatticeActivationEngine {
    fn default() -> Self {
        Self::new()
    }
}
