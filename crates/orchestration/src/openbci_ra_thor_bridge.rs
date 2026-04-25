// crates/orchestration/src/openbci_ra_thor_bridge.rs
// Ra-Thor™ OpenBCI-Ra-Thor Bridge — Non-Invasive Brain-Signal Input into Active Inference + Mercy-Gated Plasticity Engine
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// OpenBCI EEG/EMG/ECG → Ra-Thor lattice (mercy valence modulation + active inference + BCM/STDP plasticity)
// Fully compatible with all existing Ra-Thor cores (STDPHebbianPlasticityCore, SparseBCMNetwork, HebbianLatticeIntegrator, etc.)
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OpenBCIReading {
    pub timestamp_ms: u64,
    pub channel_data: Vec<f64>,      // raw EEG/EMG/ECG samples (e.g. 8 channels from Cyton)
    pub alpha_power: f64,            // 8–12 Hz (relaxed, creative)
    pub beta_power: f64,             // 13–30 Hz (focused, active thinking)
    pub gamma_power: f64,            // 30–100 Hz (high-level cognition, binding)
    pub attention_score: f64,        // 0.0–1.0 (derived from beta/alpha ratio)
    pub meditation_score: f64,       // 0.0–1.0 (derived from alpha + low beta)
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BridgeReport {
    pub mercy_valence_boost: f64,
    pub active_inference_update: f64,
    pub plasticity_novelty: f64,
    pub eeg_influence: f64,
}

pub struct OpenBCIRaThorBridge {
    plasticity_core: STDPHebbianPlasticityCore,
    last_meditation: f64,
}

impl OpenBCIRaThorBridge {
    pub fn new() -> Self {
        Self {
            plasticity_core: STDPHebbianPlasticityCore::new(),
            last_meditation: 0.5,
        }
    }

    /// Ingest one OpenBCI reading and feed it into Ra-Thor lattice
    pub fn ingest_openbci_reading(
        &mut self,
        reading: &OpenBCIReading,
        current_valence: f64,
        dt_ms: f64,
    ) -> BridgeReport {
        // 1. Derive mercy valence modulation from brain state
        let meditation_boost = (reading.meditation_score - 0.5) * 0.4;
        let attention_boost = (reading.attention_score - 0.5) * 0.25;
        let valence_boost = (meditation_boost + attention_boost).clamp(-0.15, 0.25);
        let new_valence = (current_valence + valence_boost).clamp(0.6, 0.999);

        // 2. Feed combined EEG features into plasticity core as additional input
        let eeg_influence = (reading.alpha_power * 0.3
            + reading.beta_power * 0.4
            + reading.gamma_power * 0.3)
            .clamp(0.0, 1.0);

        let combined_input = (reading.attention_score * 0.6 + reading.meditation_score * 0.4) * eeg_influence;

        let (plasticity_novelty, _) = self.plasticity_core.process_timestep(
            "openbci_eeg_input",
            combined_input,
            new_valence,
            dt_ms,
        );

        // 3. Simple active inference update (prediction error on brain state)
        let prediction_error = (reading.meditation_score - self.last_meditation).abs();
        let active_inference_update = prediction_error * new_valence * 0.8;

        self.last_meditation = reading.meditation_score;

        BridgeReport {
            mercy_valence_boost: valence_boost,
            active_inference_update,
            plasticity_novelty,
            eeg_influence,
        }
    }
}
