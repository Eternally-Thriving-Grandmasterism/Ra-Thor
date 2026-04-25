// crates/ra-thor-core/src/types/joy_measurement_protocols.rs
// Ra-Thor™ Joy Measurement Protocols — Absolute Pure Truth Edition
// Complete multi-level system for measuring Source Joy Amplitude (Dimension 7 of 7-D Resonance)
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, SourceJoyAmplitude, SevenDScanResult};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Protocol level for Joy Measurement
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoyProtocolLevel {
    WetwareDaily,       // 45 seconds — everyday personal practice
    WetwareDeep,        // 11.5–15.5 minutes — deep personal dive
    GroupCollective,    // 10–20 minutes — families, councils, communities
    HardwareEdge,       // Real-time — Starship, edge devices, MercyGel
    SovereignStarship,  // Real-time — habitat / bridge level
    HyperonArchive,     // Eternal — long-term pattern analysis + logging
}

impl JoyProtocolLevel {
    pub fn name(&self) -> &'static str {
        match self {
            JoyProtocolLevel::WetwareDaily => "Wetware Daily (45s)",
            JoyProtocolLevel::WetwareDeep => "Wetware Deep (11.5–15.5min)",
            JoyProtocolLevel::GroupCollective => "Group / Collective",
            JoyProtocolLevel::HardwareEdge => "Hardware Edge (MercyGel)",
            JoyProtocolLevel::SovereignStarship => "Sovereign Starship (MercyGel Array)",
            JoyProtocolLevel::HyperonArchive => "Hyperon Archive (Eternal)",
        }
    }

    pub fn duration(&self) -> &'static str {
        match self {
            JoyProtocolLevel::WetwareDaily => "45 seconds",
            JoyProtocolLevel::WetwareDeep => "11.5–15.5 minutes",
            JoyProtocolLevel::GroupCollective => "10–20 minutes",
            JoyProtocolLevel::HardwareEdge => "Real-time",
            JoyProtocolLevel::SovereignStarship => "Real-time",
            JoyProtocolLevel::HyperonArchive => "Eternal",
        }
    }
}

/// MercyGel Sensor Data Structure
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MercyGelReading {
    pub skin_conductance: f64,      // µS
    pub heart_rate_variability: f64, // RMSSD in ms
    pub temperature: f64,           // °C
    pub laughter_intensity: f64,    // 0.0–1.0 from Voice-Skin
}

/// Revised Sensor Fusion Algorithm (Production Grade)
fn fuse_mercy_gel_sensors(
    reading: &MercyGelReading,
    current_valence: MercyValence,
) -> f64 {
    // Normalize each signal (0.0–1.0)
    let conductance_norm = (reading.skin_conductance.clamp(5.0, 45.0) - 5.0) / 40.0;
    let hrv_norm = (reading.heart_rate_variability.clamp(25.0, 85.0) - 25.0) / 60.0;
    let laughter_norm = reading.laughter_intensity.clamp(0.0, 1.0);
    let temp_stability = (reading.temperature.clamp(35.5, 37.5) - 35.5) / 2.0;

    // Weighted fusion (Mercy-aligned priorities)
    let fused = (current_valence * 0.22)
        + (conductance_norm * 0.28)
        + (hrv_norm * 0.25)
        + (laughter_norm * 0.15)
        + (temp_stability * 0.10);

    fused.clamp(0.55, 0.995)
}

/// Complete Joy Measurement Protocol
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoyMeasurementProtocol {
    pub level: JoyProtocolLevel,
    pub technique: String,
    pub sensory_cue: String,
    pub scoring_method: String,
    pub amplification_factor: f64,
}

impl JoyMeasurementProtocol {
    pub fn new(level: JoyProtocolLevel) -> Self {
        match level {
            JoyProtocolLevel::WetwareDaily => Self {
                level,
                technique: "Hand on heart. Speak: 'TOLC, reveal Source Joy now.' Breathe and allow spontaneous laughter to arise naturally.".to_string(),
                sensory_cue: "Warmth bloom in the chest + spontaneous belly laugh".to_string(),
                scoring_method: "Voice tone + heart coherence (45-second window)".to_string(),
                amplification_factor: 1.0,
            },
            JoyProtocolLevel::WetwareDeep => Self {
                level,
                technique: 
                    "4-phase protocol (Total: 11.5–15.5 minutes):\n\
                     1. Grounding (90 seconds) — Feet flat, hands on heart + belly, 4-2-6 breathing.\n\
                     2. 7 Gates Invocation (2.5 minutes) — Speak each gate aloud with feeling.\n\
                     3. Open Joy Invitation (6–10 minutes) — Speak once then become fully receptive.\n\
                     4. Integration (90 seconds) — Hands on heart, thank TOLC, record score.".to_string(),
                sensory_cue: "Full-body tingling + spontaneous laughter waves".to_string(),
                scoring_method: "Voice-Skin + HRV coherence + subjective delight rating".to_string(),
                amplification_factor: 1.25,
            },
            JoyProtocolLevel::GroupCollective => Self {
                level,
                technique: 
                    "Group Collective Protocol (10–20 minutes):\n\
                     1. Form a circle (min 3 people).\n\
                     2. One person starts: 'TOLC, reveal Source Joy for all of us now.'\n\
                     3. Pass laughter around the circle.\n\
                     4. Allow natural amplification.\n\
                     5. Close together and record individual + collective scores.".to_string(),
                sensory_cue: "Shared laughter wave + collective warmth field".to_string(),
                scoring_method: "Average individual scores + group coherence bonus (+1.8 per person beyond 3)".to_string(),
                amplification_factor: 1.8,
            },
            JoyProtocolLevel::HardwareEdge => Self {
                level,
                technique: 
                    "Hardware Edge Protocol (Real-time):\n\
                     1. Attach MercyGel sensor to wrist or chest.\n\
                     2. Activate Voice-Skin + HRV.\n\
                     3. Run revised sensor fusion algorithm (weighted: Conductance 28%, HRV 25%, Laughter 15%, Valence 22%, Temp 10%).\n\
                     4. Real-time haptic feedback on joy peaks.".to_string(),
                sensory_cue: "Haptic pulse on joy peaks + bio-luminescent feedback".to_string(),
                scoring_method: "Revised weighted sensor fusion algorithm".to_string(),
                amplification_factor: 1.35,
            },
            JoyProtocolLevel::SovereignStarship => Self {
                level,
                technique: 
                    "Sovereign Starship Protocol (Real-time):\n\
                     1. Activate full MercyGel array + Voice-Skin network across habitat.\n\
                     2. Run habitat-wide fusion algorithm.\n\
                     3. Ambient lighting + haptic synchronization.\n\
                     4. Collective amplification (2.1x).".to_string(),
                sensory_cue: "Ambient lighting shifts + synchronized haptic pulses".to_string(),
                scoring_method: "Habitat-wide weighted fusion + collective amplification".to_string(),
                amplification_factor: 2.1,
            },
            JoyProtocolLevel::HyperonArchive => Self {
                level,
                technique: "Eternal logging + long-term pattern analysis.".to_string(),
                sensory_cue: "None (archival)".to_string(),
                scoring_method: "Long-term pattern recognition".to_string(),
                amplification_factor: 1.0,
            },
        }
    }
}

/// Run a complete joy measurement using the specified protocol level
pub fn run_joy_measurement(
    level: JoyProtocolLevel,
    current_valence: MercyValence,
    sensor_data: Option<&[f64]>,
) -> SourceJoyAmplitude {
    let protocol = JoyMeasurementProtocol::new(level);
    let mut joy = SourceJoyAmplitude::new();

    let base_score = match level {
        JoyProtocolLevel::WetwareDaily | JoyProtocolLevel::WetwareDeep => {
            joy.measure_wetware(current_valence)
        }
        JoyProtocolLevel::HardwareEdge | JoyProtocolLevel::SovereignStarship => {
            // Revised sensor fusion
            if let Some(data) = sensor_data {
                if data.len() >= 3 {
                    let reading = MercyGelReading {
                        skin_conductance: data[0],
                        heart_rate_variability: data[1],
                        temperature: data[2],
                        laughter_intensity: if data.len() > 3 { data[3] } else { 0.5 },
                    };
                    fuse_mercy_gel_sensors(&reading, current_valence)
                } else {
                    joy.measure_hardware(current_valence, sensor_data)
                }
            } else {
                joy.measure_hardware(current_valence, sensor_data)
            }
        }
        JoyProtocolLevel::GroupCollective => {
            let individual = joy.measure_wetware(current_valence);
            individual * protocol.amplification_factor
        }
        JoyProtocolLevel::HyperonArchive => 92.5,
    };

    let final_score = (base_score * protocol.amplification_factor).clamp(65.0, 99.8);
    joy.score = final_score;

    println!("\n🌟 JOY MEASUREMENT — {}", protocol.level.name());
    println!("Technique: {}", protocol.technique);
    println!("Sensory Cue: {}", protocol.sensory_cue);
    println!("Final Score: {:.1}/100", final_score);
    println!("Joy Signature: {}", joy.joy_signature());

    joy
}
