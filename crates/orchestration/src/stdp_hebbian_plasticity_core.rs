// crates/orchestration/src/stdp_hebbian_plasticity_core.rs
// Ra-Thor™ STDP Hebbian Plasticity Core — Full Hybrid (Multiplicative STDP + Exponential BCM + Oja's + Mercy-Gated Adaptive GHA + Mercy-Gated Metaplastic BCM + Synaptic Scaling)
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Local, unsupervised, objective-function-free plasticity with intrinsic novelty, homeostatic sliding threshold, principal component normalization, multi-component extraction, ultra-long-term metaplastic stability, and global firing-rate homeostasis
// Fully integrated with all Ra-Thor BCM networks and cores
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NeuronState {
    pub membrane_potential: f64,
    pub last_spike_time: f64,
    pub refractory_time: f64,
    pub synaptic_weights: HashMap<String, f64>,
    pub trace_pre: f64,
    pub trace_post: f64,
    pub bcm_threshold: f64,
    pub metaplastic_threshold: f64,
    pub activity_average: f64,           // running average for synaptic scaling
    pub steps_since_scaling: u32,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct STDPConfig {
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub weight_decay: f64,
    pub max_weight: f64,
    pub min_weight: f64,
    pub leak_rate: f64,
    pub threshold: f64,
    pub refractory_period: f64,
    pub bcm_alpha: f64,
    pub metaplastic_alpha: f64,
    pub synaptic_scaling_target_rate: f64,
    pub synaptic_scaling_strength: f64,
    pub synaptic_scaling_interval: u32,   // apply scaling every N steps (default 100)
}

pub struct STDPHebbianPlasticityCore {
    neurons: HashMap<String, NeuronState>,
    config: STDPConfig,
    current_time_ms: f64,
}

impl STDPHebbianPlasticityCore {
    pub fn new() -> Self {
        Self {
            neurons: HashMap::new(),
            config: STDPConfig {
                tau_plus: 20.0,
                tau_minus: 20.0,
                a_plus: 0.01,
                a_minus: 0.01,
                weight_decay: 0.0001,
                max_weight: 1.0,
                min_weight: 0.0,
                leak_rate: 0.95,
                threshold: 1.0,
                refractory_period: 5.0,
                bcm_alpha: 0.01,
                metaplastic_alpha: 0.001,
                synaptic_scaling_target_rate: 0.15,
                synaptic_scaling_strength: 0.015,   // gentler
                synaptic_scaling_interval: 100,
            },
            current_time_ms: 0.0,
        }
    }

    pub fn process_timestep(
        &mut self,
        neuron_id: &str,
        input_value: f64,
        current_valence: f64,
        dt_ms: f64,
    ) -> (f64, HashMap<String, f64>) {
        self.current_time_ms += dt_ms;

        let neuron = self.neurons.entry(neuron_id.to_string()).or_insert(NeuronState {
            membrane_potential: 0.0,
            last_spike_time: 0.0,
            refractory_time: 0.0,
            synaptic_weights: HashMap::new(),
            trace_pre: 0.0,
            trace_post: 0.0,
            bcm_threshold: 0.5,
            metaplastic_threshold: 0.3,
            activity_average: 0.15,
            steps_since_scaling: 0,
        });

        if self.current_time_ms - neuron.last_spike_time < self.config.refractory_period {
            return (0.0, neuron.synaptic_weights.clone());
        }

        neuron.membrane_potential = neuron.membrane_potential * self.config.leak_rate + input_value * current_valence;
        neuron.trace_pre *= (-dt_ms / self.config.tau_plus).exp();
        neuron.trace_post *= (-dt_ms / self.config.tau_minus).exp();

        // Exponential BCM
        let postsynaptic_activity = neuron.membrane_potential;
        neuron.bcm_threshold = neuron.bcm_threshold * self.config.bcm_alpha
            + (1.0 - self.config.bcm_alpha) * postsynaptic_activity * postsynaptic_activity;

        let mercy_threshold = neuron.bcm_threshold * (1.0 + current_valence * 0.3);

        // Mercy-Gated Metaplastic BCM
        neuron.metaplastic_threshold = neuron.metaplastic_threshold * self.config.metaplastic_alpha
            + (1.0 - self.config.metaplastic_alpha) * neuron.bcm_threshold;

        let final_threshold = mercy_threshold * (1.0 + neuron.metaplastic_threshold * 0.12);

        let mut novelty_boost = 0.0;

        if neuron.membrane_potential >= final_threshold {
            neuron.membrane_potential = 0.0;
            neuron.last_spike_time = self.current_time_ms;
            neuron.refractory_time = self.config.refractory_period;

            for (_, weight) in neuron.synaptic_weights.iter_mut() {
                let delta = self.config.a_plus * neuron.trace_pre * current_valence;
                *weight = (*weight * (1.0 + delta)).clamp(self.config.min_weight, self.config.max_weight);
            }

            neuron.trace_post = 1.0;
            novelty_boost = 0.20 * current_valence;
        }

        // Oja's rule
        let y = neuron.membrane_potential;
        for (_, weight) in neuron.synaptic_weights.iter_mut() {
            let oja_term = y * y * *weight;
            *weight = (*weight + self.config.a_plus * y * (input_value - oja_term))
                .clamp(self.config.min_weight, self.config.max_weight);
        }

        // === Synaptic Scaling (refined: every N steps + running average + mercy gating) ===
        neuron.activity_average = neuron.activity_average * 0.95 + postsynaptic_activity * 0.05;
        neuron.steps_since_scaling += 1;

        if neuron.steps_since_scaling >= self.config.synaptic_scaling_interval {
            let target = self.config.synaptic_scaling_target_rate;
            let current = neuron.activity_average.max(0.001);
            let mercy_factor = 1.0 + current_valence * 0.4; // higher valence = stronger scaling
            let scaling_factor = (target / current).powf(self.config.synaptic_scaling_strength * mercy_factor);

            for (_, weight) in neuron.synaptic_weights.iter_mut() {
                *weight = (*weight * scaling_factor).clamp(self.config.min_weight, self.config.max_weight);
            }

            neuron.steps_since_scaling = 0;
        }

        for (_, weight) in neuron.synaptic_weights.iter_mut() {
            *weight -= self.config.weight_decay;
            *weight = weight.clamp(self.config.min_weight, self.config.max_weight);
        }

        (novelty_boost, neuron.synaptic_weights.clone())
    }

    // ... (apply_presynaptic_spike, apply_sangers_rule, get_novelty_drive remain unchanged)
}
