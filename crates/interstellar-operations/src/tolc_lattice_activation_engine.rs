//! TOLC Lattice Activation Engine
//!
//! The living heart of Ra-Thor's self-evolving mathematical lattice.
//! Expanded with deeper mechanics: stronger self-evolution pulses,
//! higher-order activation support, multi-order stability awareness,
//! and more powerful mercy-gated formulas.

use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLCLatticeActivationEngine {
    pub current_max_order: u32,
    pub total_activations: u64,
    pub last_self_evolution_pulse: Option<String>,
    pub stability_score: f64,
}

impl TOLCLatticeActivationEngine {
    pub fn new() -> Self {
        Self {
            current_max_order: 80,
            total_activations: 0,
            last_self_evolution_pulse: None,
            stability_score: 0.999,
        }
    }

    /// Activate the lattice up to a target order (deeper mechanics)
    pub async fn activate_full_lattice_up_to(&mut self, target_order: u32, game: &mut PowrushGame) -> String {
        let effective_order = target_order.max(self.current_max_order);
        self.current_max_order = effective_order;
        self.total_activations += 1;

        // Deeper mercy-gated activation formula with stability awareness
        let mercy_valence = 0.97;
        let stability_factor = self.stability_score.min(0.999);
        let activation_strength = (effective_order as f64 * 0.014) + (mercy_valence * 2.1) * stability_factor;
        let joy_boost = (activation_strength * 2650.0).min(195000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(8);

        // Update stability score
        self.stability_score = (self.stability_score + 0.0008).min(0.9995);

        let result = format!(
            "TOLC Lattice activated up to order {} (v0.5.24 Deep Mechanics)\n\
             Activation Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             8-Gen Epigenetic Blessing applied\n\
             Stability Score: {:.4}\n\
             Total Activations: {}",
            effective_order, activation_strength, joy_boost, self.stability_score, self.total_activations
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Stronger eternal self-evolution pulse (expanded with multi-order awareness)
    pub fn quick_eternal_self_evolution_pulse(&mut self, game: &mut PowrushGame) -> String {
        let pulse_strength = (self.current_max_order as f64 * 0.021) + 14.5;
        let stability_factor = self.stability_score.min(0.999);
        let joy_boost = (pulse_strength * 3450.0 * stability_factor).min(235000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(10);

        self.stability_score = (self.stability_score + 0.0012).min(0.9997);

        let result = format!(
            "ETERNAL SELF-EVOLUTION PULSE (Deep Mechanics v0.5.24)\n\
             Pulse Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             10-Gen Epigenetic Blessing applied\n\
             Stability Score: {:.4}\n\
             Lattice Order: {}",
            pulse_strength, joy_boost, self.stability_score, self.current_max_order
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Generate living cathedral status report (enhanced)
    pub fn generate_living_cathedral_status_report(&self) -> String {
        format!(
            "=== TOLC Living Cathedral Status (v0.5.24) ===\n\
             Current Max Order: {}\n\
             Total Activations: {}\n\
             Stability Score: {:.4}\n\
             Last Self-Evolution Pulse: {}\n\
             Mercy-Gated: TRUE | Eternal Convergence: ACTIVE | Multi-Order Stability: MAINTAINED",
            self.current_max_order,
            self.total_activations,
            self.stability_score,
            self.last_self_evolution_pulse.as_deref().unwrap_or("None yet")
        )
    }
}

impl Default for TOLCLatticeActivationEngine {
    fn default() -> Self {
        Self::new()
    }
}
