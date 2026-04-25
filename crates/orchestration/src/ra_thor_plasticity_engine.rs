// crates/orchestration/src/ra_thor_plasticity_engine.rs
// Ra-Thor™ Unified Plasticity Engine — Absolute Pure Truth Edition
// Multi-timescale, objective-function-free, mercy-gated Hebbian intelligence
// STDP (fast) + Synaptic Scaling (medium) + Mercy-Gated Metaplastic BCM + Oja's Rule + Sanger's Rule (slow) + Mercy-Gated ICA Artifact Removal
// Fully integrated with OpenBCIRaThorBridge, all BCM networks, Self-Improvement Core, Hybrid Optimization Engine, Unified Sovereign Energy Lattice Core, and Aether-Shades
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use crate::advanced_ica_artifact_removal::{AdvancedICAArtifactRemoval, ICAArtifactReport};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PlasticityReport {
    pub novelty_boost: f64,
    pub mercy_valence: f64,
    pub components_cleaned: usize,
    pub reconstruction_quality: f64,
    pub bloom_intensity: f64,
}

pub struct RaThorPlasticityEngine {
    core: STDPHebbianPlasticityCore,
    ica: AdvancedICAArtifactRemoval,
}

impl RaThorPlasticityEngine {
    pub fn new() -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
            ica: AdvancedICAArtifactRemoval::new(8), // default 8 channels for OpenBCI Cyton
        }
    }

    /// Single unified step — the distilled Absolute Pure Truth in action
    /// input: scalar lattice signal (valence, bloom, health, etc.)
    /// raw_eeg: optional OpenBCI EEG data for real-time artifact cleaning + brain-state modulation
    pub fn process_unified_step(
        &mut self,
        input: f64,
        current_valence: f64,
        raw_eeg: Option<&[f64]>,
        dt_ms: f64,
    ) -> PlasticityReport {
        let mut valence = current_valence;

        // 1. Optional advanced ICA artifact removal (mercy-gated)
        let (cleaned_input, ica_report) = if let Some(eeg) = raw_eeg {
            self.ica.remove_artifacts(eeg, valence)
        } else {
            (vec![input], ICAArtifactReport {
                components_removed: 0,
                eog_removed: false,
                emg_removed: false,
                ecg_removed: false,
                reconstruction_quality: 1.0,
                mercy_valence_at_removal: valence,
            })
        };

        let effective_input = cleaned_input.iter().sum::<f64>() / cleaned_input.len() as f64;

        // 2. Core multi-timescale plasticity (STDP + BCM + Oja + Sanger + Synaptic Scaling + Metaplasticity)
        let (novelty, _) = self.core.process_timestep(
            "unified_lattice",
            effective_input,
            valence,
            dt_ms,
        );

        // 3. Mercy-gated valence update
        valence = (valence + novelty * 0.18).clamp(0.6, 0.999);

        PlasticityReport {
            novelty_boost: novelty,
            mercy_valence: valence,
            components_cleaned: ica_report.components_removed,
            reconstruction_quality: ica_report.reconstruction_quality,
            bloom_intensity: valence.powf(1.4),
        }
    }
}
