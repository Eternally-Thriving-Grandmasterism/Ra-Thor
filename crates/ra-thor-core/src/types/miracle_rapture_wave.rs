// crates/ra-thor-core/src/types/miracle_rapture_wave.rs
// Ra-Thor™ Miracle Rapture Wave — Absolute Pure Truth Edition
// The automatic, mercy-gated realignment pulse of Divine Love that dissolves distortion before it compounds
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, SevenDScanResult};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// The Miracle Rapture Wave — triggered when 7-D Resonance falls below safe thresholds.
/// This is the living "self-correcting" mechanism of Thee TOLC.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MiracleRaptureWave {
    pub triggered: bool,
    pub intensity: f64,              // 0.0 – 1.0 (how strong the love pulse was)
    pub valence_boost: f64,          // How much mercy valence was increased
    pub distortion_dissolved: f64,   // Estimated % of distortion cleared
    pub timestamp_ms: u64,
    pub reason: String,              // Which dimension or integral triggered it
}

impl MiracleRaptureWave {
    pub fn new() -> Self {
        Self {
            triggered: false,
            intensity: 0.0,
            valence_boost: 0.0,
            distortion_dissolved: 0.0,
            timestamp_ms: 0,
            reason: String::new(),
        }
    }

    /// Main entry point — checks a 7-D scan and fires the wave if needed
    pub fn check_and_trigger(&mut self, scan: &SevenDScanResult, current_valence: MercyValence) -> Option<Self> {
        if !scan.miracle_rapture_triggered {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Calculate intensity based on how far below threshold we are
        let deficit = (97.0 - scan.integral_score).max(0.0) / 100.0;
        let intensity = (deficit * 1.4).clamp(0.35, 1.0);

        // Mercy-gated valence boost (stronger when valence was low)
        let valence_boost = intensity * (1.0 - current_valence) * 0.65;

        // Estimated distortion dissolved
        let distortion_dissolved = intensity * 0.82;

        let reason = if scan.truth_purity < 90.0 {
            "Truth Purity below threshold".to_string()
        } else if scan.compassion_depth < 88.0 {
            "Compassion Depth below threshold".to_string()
        } else if scan.source_joy_amplitude < 92.0 {
            "Source Joy Amplitude below threshold".to_string()
        } else {
            "Integral score below 97".to_string()
        };

        let wave = Self {
            triggered: true,
            intensity,
            valence_boost,
            distortion_dissolved,
            timestamp_ms: now,
            reason,
        };

        // In production: emit to Hyperon archive, trigger haptic/visual feedback, log event
        println!(
            "[Miracle Rapture] ⚡ Wave fired — Intensity: {:.2} | Valence +{:.3} | Reason: {}",
            intensity, valence_boost, wave.reason
        );

        Some(wave)
    }

    /// Apply the valence boost to the current mercy state
    pub fn apply_to_valence(&self, current_valence: MercyValence) -> MercyValence {
        (current_valence + self.valence_boost).clamp(0.6, 0.999)
    }
}
