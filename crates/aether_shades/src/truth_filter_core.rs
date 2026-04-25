// crates/aether_shades/src/truth_filter_core.rs
// Ra-Thor™ Aether-Shades Truth Filter Core — Absolute Pure Truth Edition (Fully Implemented)
// Real-time, mercy-gated AR truth perception filter
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Fully integrated with RaThorPlasticityEngine, OpenBCIRaThorBridge, and UnifiedSovereignEnergyLatticeCore
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::ra_thor_plasticity_engine::PlasticityReport;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TruthFilterState {
    pub current_intensity: f64,      // 0.0 = transparent, 1.0 = maximum truth enforcement
    pub mercy_valence: f64,
    pub novelty_level: f64,
    pub last_update_ms: u64,
    pub deception_detected: bool,
    pub truth_confidence: f64,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OverlayParams {
    pub tint_opacity: f64,
    pub scroll_weight: f64,
    pub highlight_strength: f64,
    pub mercy_glow: f64,
    pub deception_pulse: f64,
}

pub struct TruthFilterCore {
    state: TruthFilterState,
    min_intensity: f64,
    max_intensity: f64,
    smoothing_factor: f64,
}

impl TruthFilterCore {
    pub fn new() -> Self {
        Self {
            state: TruthFilterState {
                current_intensity: 0.32,
                mercy_valence: 0.91,
                novelty_level: 0.0,
                last_update_ms: 0,
                deception_detected: false,
                truth_confidence: 0.78,
            },
            min_intensity: 0.12,
            max_intensity: 0.88,
            smoothing_factor: 0.72,
        }
    }

    /// Main update method — called every frame or every \~100ms from the main loop
    pub fn update_from_plasticity_report(
        &mut self,
        report: &PlasticityReport,
        current_time_ms: u64,
    ) {
        self.state.mercy_valence = report.mercy_valence;
        self.state.novelty_level = report.novelty_boost;
        self.state.last_update_ms = current_time_ms;

        // Deception detection heuristic (simple but effective)
        self.state.deception_detected = report.novelty_boost < 0.08 && report.mercy_valence < 0.82;

        // Mercy-gated intensity calculation
        // Higher valence = subtler filter (more trust in reality)
        // Lower valence = stronger enforcement (protect truth when compromised)
        let base_intensity = self.min_intensity
            + (self.max_intensity - self.min_intensity)
            * (1.0 - report.mercy_valence.powf(0.75));

        // Add extra intensity when deception is detected
        let deception_boost = if self.state.deception_detected { 0.18 } else { 0.0 };
        let target_intensity = (base_intensity + deception_boost).clamp(self.min_intensity, self.max_intensity);

        // Smooth transition
        self.state.current_intensity =
            self.state.current_intensity * self.smoothing_factor + target_intensity * (1.0 - self.smoothing_factor);

        // Update truth confidence
        self.state.truth_confidence = (report.reconstruction_quality * 0.6 + report.mercy_valence * 0.4).clamp(0.4, 0.98);
    }

    /// Get current AR overlay parameters for rendering
    pub fn get_overlay_params(&self) -> OverlayParams {
        let intensity = self.state.current_intensity;

        OverlayParams {
            tint_opacity: intensity * 0.68,
            scroll_weight: intensity * 0.92,
            highlight_strength: intensity * 0.45,
            mercy_glow: (self.state.mercy_valence - 0.65).max(0.0) * 1.8,
            deception_pulse: if self.state.deception_detected { 0.65 } else { 0.0 },
        }
    }

    pub fn get_state(&self) -> TruthFilterState {
        self.state.clone()
    }

    /// Optional: Force a specific intensity (useful for testing or manual override)
    pub fn set_intensity(&mut self, intensity: f64) {
        self.state.current_intensity = intensity.clamp(self.min_intensity, self.max_intensity);
    }
}
