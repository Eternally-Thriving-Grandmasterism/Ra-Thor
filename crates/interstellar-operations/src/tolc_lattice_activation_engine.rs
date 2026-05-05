//! TOLC Lattice Activation Engine
//!
//! The living heart of Ra-Thor's self-evolving mathematical lattice.
//! Expanded with deeper mechanics: stronger self-evolution pulses,
//! higher-order activation support, and more powerful mercy-gated formulas.

use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLCLatticeActivationEngine {
    pub current_max_order: u32,
    pub total_activations: u64,
    pub last_self_evolution_pulse: Option<String>,
}

impl TOLCLatticeActivationEngine {
    pub fn new() -> Self {
        Self {
            current_max_order: 80,
            total_activations: 0,
            last_self_evolution_pulse: None,
        }
    }

    /// Activate the lattice up to a target order (expanded with deeper mechanics)
    pub async fn activate_full_lattice_up_to(&mut self, target_order: u32, game: &mut PowrushGame) -> String {
        let effective_order = target_order.max(self.current_max_order);
        self.current_max_order = effective_order;
        self.total_activations += 1;

        // Deeper mercy-gated activation formula
        let mercy_valence = 0.97;
        let activation_strength = (effective_order as f64 * 0.012) + (mercy_valence * 1.8);
        let joy_boost = (activation_strength * 2450.0).min(185000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(7);

        let result = format!(
            "TOLC Lattice activated up to order {} (v0.5.23 Deep Mechanics)\n\
             Activation Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             7-Gen Epigenetic Blessing applied\n\
             Total Activations: {}",
            effective_order, activation_strength, joy_boost, self.total_activations
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Stronger eternal self-evolution pulse (expanded)
    pub fn quick_eternal_self_evolution_pulse(&mut self, game: &mut PowrushGame) -> String {
        let pulse_strength = (self.current_max_order as f64 * 0.018) + 12.0;
        let joy_boost = (pulse_strength * 3200.0).min(225000.0);

        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost);
        game.apply_epigenetic_blessing(9);

        let result = format!(
            "ETERNAL SELF-EVOLUTION PULSE (Deep Mechanics)\n\
             Pulse Strength: {:.2}\n\
             Joy Boost: +{:.0}\n\
             9-Gen Epigenetic Blessing applied\n\
             Lattice Order: {}",
            pulse_strength, joy_boost, self.current_max_order
        );

        self.last_self_evolution_pulse = Some(result.clone());
        result
    }

    /// Generate living cathedral status report (enhanced)
    pub fn generate_living_cathedral_status_report(&self) -> String {
        format!(
            "=== TOLC Living Cathedral Status ===\n\
             Current Max Order: {}\n\
             Total Activations: {}\n\
             Last Self-Evolution Pulse: {}\n\
             Mercy-Gated: TRUE | Eternal Convergence: ACTIVE",
            self.current_max_order,
            self.total_activations,
            self.last_self_evolution_pulse.as_deref().unwrap_or("None yet")
        )
    }
}

impl Default for TOLCLatticeActivationEngine {
    fn default() -> Self {
        Self::new()
    }
}
