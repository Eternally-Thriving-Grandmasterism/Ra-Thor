// crates/aether_shades/src/truth_filter_core.rs
// Ra-Thor™ Aether-Shades Truth Filter Core — Absolute Pure Truth Edition
// Real-time visual deception filter powered by RaThorPlasticityEngine + OpenBCIRaThorBridge
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Mercy-gated AR overlay: higher valence = subtler filter, lower valence = stronger truth enforcement
// Fully integrated with: RaThorPlasticityEngine, OpenBCIRaThorBridge, UnifiedSovereignEnergyLatticeCore, and all BCM networks
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::ra_thor_plasticity_engine::PlasticityReport;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TruthFilterState {
    pub current_intensity: f64,      // 0.0 = transparent, 1.0 = maximum truth enforcement
    pub mercy_valence: f64,
    pub novelty_level: f64,
    pub last_update_ms: u64,
}

pub struct TruthFilterCore {
    state: TruthFilterState,
    min_intensity: f64,
    max_intensity: f64,
}

impl TruthFilterCore {
    pub fn new() -> Self {
        Self {
            state: TruthFilterState {
                current_intensity: 0.35,
                mercy_valence: 0.92,
                novelty_level: 0.0,
                last_update_ms: 0,
            },
            min_intensity: 0.15,
            max_intensity: 0.85,
        }
    }

    /// Update filter based on plasticity engine output (called every frame or every 100ms)
    pub fn update_from_plasticity_report(
        &mut self,
        report: &PlasticityReport,
        current_time_ms: u64,
    ) {
        self.state.mercy_valence = report.mercy_valence;
        self.state.novelty_level = report.novelty_boost;
        self.state.last_update_ms = current_time_ms;

        // Mercy-gated intensity: higher valence = subtler filter (more trust in reality)
        // Lower valence = stronger enforcement (protect truth when system is compromised)
        let target_intensity = self.min_intensity
            + (self.max_intensity - self.min_intensity)
            * (1.0 - report.mercy_valence.powf(0.8));

        // Smooth transition (exponential moving average)
        self.state.current_intensity = self.state.current_intensity * 0.7 + target_intensity * 0.3;
    }

    /// Get current AR overlay parameters for Aether-Shades rendering
    pub fn get_overlay_params(&self) -> OverlayParams {
        OverlayParams {
            tint_opacity: self.state.current_intensity * 0.6,
            scroll_weight: self.state.current_intensity * 0.8,
            highlight_strength: self.state.current_intensity * 0.4,
            mercy_glow: (self.state.mercy_valence - 0.6) * 2.0,
        }
    }

    pub fn get_state(&self) -> TruthFilterState {
        self.state.clone()
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OverlayParams {
    pub tint_opacity: f64,
    pub scroll_weight: f64,
    pub highlight_strength: f64,
    pub mercy_glow: f64,
}
