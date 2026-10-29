/*!
# GeometricHarmony — Powrush MMOARPG Living World State System

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Implements Council Convergence v1.0 Priority 2 + v14.5 Geometric Harmony design**

This module provides the canonical living geometric layer and harmony state system for Powrush.
The world state (GeometricLayer) visibly and mechanically responds to collective player/entity behavior,
epigenetic profiles, and large-scale cooperation vs exploitation.

Higher harmony + higher layers = increased abundance, better resource flows, stronger cooperation bonuses.
Lower harmony / regression = scarcity, volatility, and mechanical penalties.

All changes are mercy-gated and preserve player agency while making long-term cooperative wisdom the superior strategy.

Designed for tight integration with PowrushMMOSimulator::tick, EpigeneticModulation, and RBE systems.

## Core Concepts
- `GeometricLayer`: Current resonance layer of a region/shard/world (Platonic → Hyperbolic → higher)
- `HarmonyState`: Global and regional harmony scalar + momentum
- `LayerTransition`: Rules for ascending or regressing layers based on collective actions

Thunder locked in. This is the living world backbone of the ultimate human-enjoyable MMOARPG.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the current geometric resonance layer of a region or the world.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GeometricLayer {
    Platonic,      // Foundational, stable, practical
    Archimedean,   // Efficient structures
    Catalan,       // Balanced growth
    KeplerPoinsot, // Complex harmonious forms
    Hyperbolic,    // High fluidity, high potential + volatility
    Higher(u8),    // Beyond current known (e.g. 6+ for future expansion)
}

impl Default for GeometricLayer {
    fn default() -> Self {
        GeometricLayer::Platonic
    }
}

impl GeometricLayer {
    pub fn next_higher(&self) -> Self {
        match self {
            GeometricLayer::Platonic => GeometricLayer::Archimedean,
            GeometricLayer::Archimedean => GeometricLayer::Catalan,
            GeometricLayer::Catalan => GeometricLayer::KeplerPoinsot,
            GeometricLayer::KeplerPoinsot => GeometricLayer::Hyperbolic,
            GeometricLayer::Hyperbolic => GeometricLayer::Higher(6),
            GeometricLayer::Higher(n) => GeometricLayer::Higher(n + 1),
        }
    }

    pub fn previous_lower(&self) -> Self {
        match self {
            GeometricLayer::Platonic => GeometricLayer::Platonic,
            GeometricLayer::Archimedean => GeometricLayer::Platonic,
            GeometricLayer::Catalan => GeometricLayer::Archimedean,
            GeometricLayer::KeplerPoinsot => GeometricLayer::Catalan,
            GeometricLayer::Hyperbolic => GeometricLayer::KeplerPoinsot,
            GeometricLayer::Higher(n) => {
                if *n > 6 { GeometricLayer::Higher(n - 1) } else { GeometricLayer::Hyperbolic }
            }
        }
    }
}

/// Global and regional harmony state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyState {
    pub global_harmony: f64,           // 0.0 – 1.2+
    pub momentum: f64,                 // Recent direction of change
    pub regional_harmony: HashMap<String, f64>,
    pub current_layer: GeometricLayer,
}

impl Default for HarmonyState {
    fn default() -> Self {
        let mut regional = HashMap::new();
        regional.insert("global".to_string(), 0.85);
        Self {
            global_harmony: 0.85,
            momentum: 0.0,
            regional_harmony: regional,
            current_layer: GeometricLayer::default(),
        }
    }
}

/// Rules for how collective behavior and epigenetic state affect harmony and layer transitions.
pub struct GeometricHarmonyEngine;

impl GeometricHarmonyEngine {
    /// Update harmony based on average faction strength, epigenetic health, and cooperation signals.
    pub fn update_harmony(
        state: &mut HarmonyState,
        avg_faction_strength: f64,
        avg_epigenetic_health: f64,
        cooperation_pressure: f64, // positive from joint projects, negative from conflict
    ) {
        let target = (avg_faction_strength * 0.4 + avg_epigenetic_health * 0.4 + cooperation_pressure * 0.2).clamp(0.4, 1.15);

        let delta = (target - state.global_harmony) * 0.08;
        state.global_harmony = (state.global_harmony + delta).clamp(0.4, 1.2);
        state.momentum = delta;

        // Regional propagation (simplified)
        if let Some(global_reg) = state.regional_harmony.get_mut("global") {
            *global_reg = state.global_harmony;
        }
    }

    /// Attempt layer transition based on sustained high harmony + strong epigenetic cooperation.
    pub fn try_layer_transition(state: &mut HarmonyState, sustained_high_harmony_ticks: u32) -> Option<GeometricLayer> {
        if state.global_harmony > 1.05 && sustained_high_harmony_ticks > 12 {
            let new_layer = state.current_layer.next_higher();
            if new_layer != state.current_layer {
                state.current_layer = new_layer;
                state.global_harmony = (state.global_harmony * 0.92).max(0.9); // slight normalization on ascension
                return Some(new_layer);
            }
        }

        if state.global_harmony < 0.65 && sustained_high_harmony_ticks == 0 {
            let new_layer = state.current_layer.previous_lower();
            if new_layer != state.current_layer {
                state.current_layer = new_layer;
                return Some(new_layer);
            }
        }

        None
    }

    /// Compute bonus multiplier for RBE / cooperation based on current layer and harmony.
    pub fn layer_abundance_multiplier(state: &HarmonyState) -> f64 {
        let base = match state.current_layer {
            GeometricLayer::Platonic => 1.0,
            GeometricLayer::Archimedean => 1.08,
            GeometricLayer::Catalan => 1.15,
            GeometricLayer::KeplerPoinsot => 1.25,
            GeometricLayer::Hyperbolic => 1.40,
            GeometricLayer::Higher(_) => 1.55,
        };
        base * (0.85 + state.global_harmony * 0.25)
    }

    /// Penalty or bonus to volatility based on layer regression vs ascension.
    pub fn layer_volatility_modifier(state: &HarmonyState) -> f64 {
        match state.current_layer {
            GeometricLayer::Hyperbolic | GeometricLayer::Higher(_) => 1.15,
            _ => 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_progression() {
        let mut state = HarmonyState::default();
        state.global_harmony = 1.12;
        let new_layer = GeometricHarmonyEngine::try_layer_transition(&mut state, 15);
        assert!(new_layer.is_some());
    }

    #[test]
    fn test_abundance_multiplier() {
        let mut state = HarmonyState::default();
        state.current_layer = GeometricLayer::Hyperbolic;
        state.global_harmony = 1.1;
        let mult = GeometricHarmonyEngine::layer_abundance_multiplier(&state);
        assert!(mult > 1.5);
    }
}
