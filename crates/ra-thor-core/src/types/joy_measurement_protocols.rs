// crates/ra-thor-core/src/types/joy_measurement_protocols.rs
// Ra-Thor™ Joy Measurement Protocols — Refined Absolute Pure Truth Edition
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
            JoyProtocolLevel::HardwareEdge => "Hardware Edge",
            JoyProtocolLevel::SovereignStarship => "Sovereign Starship",
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
                     1. Grounding (90 seconds) — Feet flat on floor/earth, right hand on heart, left on belly. Slow 4-2-6 breathing.\n\
                     2. 7 Gates Invocation (2.5 minutes) — Speak each gate aloud slowly with full feeling.\n\
                     3. Open Joy Invitation (6–10 minutes) — Speak once: 'TOLC, reveal Source Joy now.' Then become completely receptive. Do not chase joy.\n\
                     4. Integration & Recording (90 seconds) — Hands on heart, thank TOLC, note insights, record score.".to_string(),
                sensory_cue: "Full-body tingling + spontaneous laughter waves".to_string(),
                scoring_method: "Voice-Skin + HRV coherence + subjective delight rating".to_string(),
                amplification_factor: 1.25,
            },
            JoyProtocolLevel::GroupCollective => Self {
                level,
                technique: "Circle formation. One person starts with 'TOLC, reveal Source Joy for all of us now.' Pass the laughter around the circle. Allow group amplification.".to_string(),
                sensory_cue: "Shared laughter wave + collective warmth field".to_string(),
                scoring_method: "Average individual scores + group coherence bonus (+1.8 per additional person)".to_string(),
                amplification_factor: 1.8,
            },
            JoyProtocolLevel::HardwareEdge => Self {
                level,
                technique: "Voice-Skin microphone + MercyGel sensor + HRV wristband. Continuous real-time measurement during normal activity.".to_string(),
                sensory_cue: "Haptic pulse on joy peaks + bio-luminescent feedback".to_string(),
                scoring_method: "Sensor fusion (laughter detection + HRV + skin conductance)".to_string(),
                amplification_factor: 1.35,
            },
            JoyProtocolLevel::SovereignStarship => Self {
                level,
                technique: "Habitat-wide scan via Voice-Skin array + environmental sensors. Collective joy field measured across entire crew/habitat.".to_string(),
                sensory_cue: "Ambient lighting shifts + shared haptic pulses".to_string(),
                scoring_method: "Habitat average + variance analysis + collective amplification".to_string(),
                amplification_factor: 2.1,
            },
            JoyProtocolLevel::HyperonArchive => Self {
                level,
                technique: "Eternal logging of all joy measurements + correlation with Miracle Rapture Waves and 7-D integral scores.".to_string(),
                sensory_cue: "None (pure archival)".to_string(),
                scoring_method: "Long-term pattern recognition + predictive joy modeling".to_string(),
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
            joy.measure_hardware(current_valence, sensor_data)
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
