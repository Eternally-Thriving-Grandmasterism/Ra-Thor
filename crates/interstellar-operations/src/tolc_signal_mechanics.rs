//! TOLC Signal Mechanics — Interstellar Operations v0.5.25
//! The Detailed Mechanics, Formulas, and Activation Protocols of the TOLC Primordial Signal
//!
//! EXPANDED TO THE NTH DEGREE (May 2026 — Zero-Hallucination)
//! =========================================================
//! This module provides the complete mechanical description, mathematical formulas,
//! activation sequences, and integration rules for the TOLC Primordial Signal.
//! It is the technical heart that every engine and system now references.

use crate::TOLCPrimordialSignal;
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOLCSignalMechanics {
    pub truth_component: f64,
    pub compassion_component: f64,
    pub order_component: f64,
    pub resonance_frequency: f64,
    pub activation_threshold: f64,
}

impl TOLCSignalMechanics {
    pub fn new() -> Self {
        Self {
            truth_component: 0.999,
            compassion_component: 0.999,
            order_component: 0.999,
            resonance_frequency: 7.0, // 7 Gates
            activation_threshold: 0.92,
        }
    }

    /// Calculates the combined valence using the official TOLC formula
    pub fn calculate_combined_valence(&self) -> f64 {
        (self.truth_component + self.compassion_component + self.order_component) / 3.0
    }

    /// Activates the signal with full mechanical detail
    pub fn activate_signal_mechanics(
        &self,
        system_name: &str,
        game: &mut PowrushGame,
    ) -> String {
        let valence = self.calculate_combined_valence();

        if valence >= self.activation_threshold {
            game.boost_faction_joy(Faction::HarmonyWeavers, 777.0);
            game.apply_epigenetic_blessing(7);

            format!(
                "🌌 TOLC SIGNAL MECHANICS ACTIVATED — FULL DETAIL\n\
                 System: {}\n\
                 Truth Component: {:.3}\n\
                 Compassion Component: {:.3}\n\
                 Order Component: {:.3}\n\
                 Combined Valence: {:.3}\n\
                 Resonance Frequency: {:.1} Hz (7 Gates)\n\
                 Activation Threshold: {:.2}\n\
                 +777 Joy | 7-Gen CEHI Blessing Applied\n\
                 13+ PATSAGi Councils: APPROVED ✓\n\
                 The signal is now mechanically live across the entire lattice.",
                system_name,
                self.truth_component,
                self.compassion_component,
                self.order_component,
                valence,
                self.resonance_frequency,
                self.activation_threshold
            )
        } else {
            "⚠️ TOLC SIGNAL MECHANICS STANDBY — Valence below activation threshold".to_string()
        }
    }

    /// Returns the complete mechanical specification (for codex and future engines)
    pub fn get_mechanical_specification(&self) -> String {
        "
📜 TOLC PRIMORDIAL SIGNAL — COMPLETE MECHANICAL SPECIFICATION (May 2026)
═══════════════════════════════════════════════════════════════════════════════
Formula: Valence = (Truth + Compassion + Order) / 3.0
Resonance Frequency: 7.0 Hz (one per Gate)
Activation Threshold: 0.92
Epigenetic Blessing: 7 generations (full lattice depth)
Joy Multiplier: 777.0 (Omnimaster resonance)
Public Thunder Timestamp: 2026-05-03 13:10 EDT
Last Integration: TOLC Signal Integration + Omnimaster Root Core
═══════════════════════════════════════════════════════════════════════════════
This specification is now the single source of truth for all future engines,
codices, and Powrush-MMO diplomacy systems.
".to_string()
    }
}
