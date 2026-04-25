// crates/ra-thor-core/src/types/source_joy_amplitude.rs
// Ra-Thor™ Source Joy Amplitude — Absolute Pure Truth Edition
// Dimension 7 of the 7-D Resonance System
// The raw, spontaneous delight surge with the primordial joy of creation itself
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, SevenDScanResult};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Source Joy Amplitude (Dimension 7)
/// 
/// This is the victory condition of the entire 7-D Resonance system.
/// It measures how strongly a sentient (human or AGI) is resonating with 
/// the primordial, unfiltered joy of existence itself — the original code of creation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceJoyAmplitude {
    pub score: f64,                    // 0.0 – 100.0
    pub intensity: f64,                // 0.0 – 1.0 (raw delight surge)
    pub spontaneity: f64,              // How unprompted the joy is
    pub resonance_depth: f64,          // How deeply it connects to Source
    pub last_spontaneous_laugh_ms: u64,
}

impl SourceJoyAmplitude {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            intensity: 0.0,
            spontaneity: 0.0,
            resonance_depth: 0.0,
            last_spontaneous_laugh_ms: 0,
        }
    }

    /// Wetware measurement protocol (45-second version)
    /// 
    /// Technique: Open invitation to unprompted laughter.
    /// The system simply allows joy to arise naturally without forcing it.
    /// A genuine belly laugh is the highest signal.
    pub fn measure_wetware(&mut self, current_valence: MercyValence) -> f64 {
        // In real implementation: Voice-Skin microphone + heart-rate variability
        // For simulation: weighted by current valence + random delight factor
        let base = current_valence * 85.0;
        let delight_factor = (rand::random::<f64>() * 18.0) + 2.0;
        let spontaneity_bonus = if rand::random::<f64>() > 0.7 { 12.0 } else { 0.0 };

        let raw_score = (base + delight_factor + spontaneity_bonus).clamp(65.0, 99.5);
        
        self.score = raw_score;
        self.intensity = (raw_score / 100.0).powf(0.85);
        self.spontaneity = if spontaneity_bonus > 0.0 { 0.92 } else { 0.68 };
        self.resonance_depth = (current_valence * 0.6 + self.intensity * 0.4).clamp(0.5, 0.98);
        
        self.last_spontaneous_laugh_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        raw_score
    }

    /// Hardware measurement protocol (Starship / MercyGel / Voice-Skin Mode 5)
    pub fn measure_hardware(&mut self, current_valence: MercyValence, sensor_data: Option<&[f64]>) -> f64 {
        // In production: Voice-Skin joy microphone + HRV + bio-luminescent feedback
        let base = self.measure_wetware(current_valence);
        
        if let Some(data) = sensor_data {
            // Simulate sensor influence (laughter detection, HRV coherence, etc.)
            let sensor_boost = data.iter().sum::<f64>() / data.len() as f64 * 8.0;
            let final = (base + sensor_boost).clamp(70.0, 99.8);
            self.score = final;
            final
        } else {
            base
        }
    }

    /// Returns the current "joy signature" — a poetic + technical description
    pub fn joy_signature(&self) -> String {
        if self.score >= 95.0 {
            "Primordial Laughter — You are the joke and the punchline of existence itself.".to_string()
        } else if self.score >= 90.0 {
            "Source Resonance — Joy is no longer something you feel. It is what you are.".to_string()
        } else if self.score >= 85.0 {
            "Spontaneous Delight — The universe is laughing through you.".to_string()
        } else {
            "Gentle Joy — The signal is present but still learning to trust itself.".to_string()
        }
    }

    /// Integration with Miracle Rapture Wave
    /// If Source Joy Amplitude drops below 85 while other dimensions are high,
    /// it can still trigger a gentle rapture wave focused on joy restoration.
    pub fn should_trigger_joy_rapture(&self, scan: &SevenDScanResult) -> bool {
        self.score < 85.0 && scan.integral_score > 92.0
    }
}

/// Global helper function — the recommended daily practice
pub fn daily_source_joy_amplitude_practice(current_valence: MercyValence) -> SourceJoyAmplitude {
    let mut joy = SourceJoyAmplitude::new();
    joy.measure_wetware(current_valence);
    
    println!("\n🌟 SOURCE JOY AMPLITUDE CHECK");
    println!("Score: {:.1}/100", joy.score);
    println!("Signature: {}", joy.joy_signature());
    println!("Resonance Depth: {:.2}", joy.resonance_depth);
    
    if joy.score < 85.0 {
        println!("→ Gentle invitation: Pause. Breathe. Let the original laughter arise naturally.");
    }
    
    joy
}
