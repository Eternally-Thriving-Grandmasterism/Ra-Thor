// crates/orchestration/src/stdp_hebbian_plasticity_core.rs
// Ra-Thor™ STDP Hebbian Plasticity Core — Multiplicative STDP + Exponential BCM + Oja's Rule + Sanger's Rule (Full Hybrid)
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Local, unsupervised, objective-function-free plasticity with intrinsic novelty, homeostatic sliding threshold, principal component normalization, and multi-component extraction
// Fully integrated with HebbianNoveltyCore, Self-Improvement Core, Hybrid Optimization, Unified Lattice
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
        });

        if self.current_time_ms - neuron.last_spike_time < self.config.refractory_period {
            return (0.0, neuron.synaptic_weights.clone());
        }

        neuron.membrane_potential = neuron.membrane_potential * self.config.leak_rate + input_value * current_valence;
        neuron.trace_pre *= (-dt_ms / self.config.tau_plus).exp();
        neuron.trace_post *= (-dt_ms / self.config.tau_minus).exp();

        let postsynaptic_activity = neuron.membrane_potential;
        neuron.bcm_threshold = neuron.bcm_threshold * self.config.bcm_alpha
            + (1.0 - self.config.bcm_alpha) * postsynaptic_activity * postsynaptic_activity;

        let mercy_threshold = neuron.bcm_threshold * (1.0 + current_valence * 0.3);

        let mut novelty_boost = 0.0;

        if neuron.membrane_potential >= mercy_threshold {
            neuron.membrane_potential = 0.0;
            neuron.last_spike_time = self.current_time_ms;
            neuron.refractory_time = self.config.refractory_period;

            for (_, weight) in neuron.synaptic_weights.iter_mut() {
                let delta = self.config.a_plus * neuron.trace_pre * current_valence;
                *weight = (*weight * (1.0 + delta)).clamp(self.config.min_weight, self.config.max_weight);
            }

            neuron.trace_post = 1.0;
            novelty_boost = 0.18 * current_valence;
        }

        // Oja's rule normalization
        let y = neuron.membrane_potential;
        for (_, weight) in neuron.synaptic_weights.iter_mut() {
            let oja_term = y * y * *weight;
            *weight = (*weight + self.config.a_plus * y * (input_value - oja_term))
                .clamp(self.config.min_weight, self.config.max_weight);
        }

        for (_, weight) in neuron.synaptic_weights.iter_mut() {
            *weight -= self.config.weight_decay;
            *weight = weight.clamp(self.config.min_weight, self.config.max_weight);
        }

        (novelty_boost, neuron.synaptic_weights.clone())
    }

    pub fn apply_presynaptic_spike(&mut self, neuron_id: &str, pre_neuron_id: &str, dt_ms: f64) {
        let neuron = self.neurons.entry(neuron_id.to_string()).or_insert(NeuronState {
            membrane_potential: 0.0,
            last_spike_time: 0.0,
            refractory_time: 0.0,
            synaptic_weights: HashMap::new(),
            trace_pre: 0.0,
            trace_post: 0.0,
            bcm_threshold: 0.5,
        });

        let weight = neuron.synaptic_weights.entry(pre_neuron_id.to_string()).or_insert(0.5);

        if self.current_time_ms - neuron.last_spike_time > 0.0 && self.current_time_ms - neuron.last_spike_time < 50.0 {
            let delta = self.config.a_minus * neuron.trace_post * (-dt_ms / self.config.tau_minus).exp();
            *weight = (*weight * (1.0 - delta)).clamp(self.config.min_weight, self.config.max_weight);
        }

        neuron.trace_pre = 1.0;
    }

    /// Sanger's Rule (Generalized Hebbian Algorithm) — multi-component principal component extraction
    pub fn apply_sangers_rule(
        &mut self,
        neuron_id: &str,
        input_vector: &[f64],
        component_index: usize,
        current_valence: f64,
    ) {
        let neuron = self.neurons.entry(neuron_id.to_string()).or_insert(NeuronState {
            membrane_potential: 0.0,
            last_spike_time: 0.0,
            refractory_time: 0.0,
            synaptic_weights: HashMap::new(),
            trace_pre: 0.0,
            trace_post: 0.0,
            bcm_threshold: 0.5,
        });

        let y = neuron.membrane_potential;

        // Deflation: subtract projections of previous components
        let mut deflation = 0.0;
        for j in 0..component_index {
            let prev_id = format!("{}_pc{}", neuron_id, j);
            if let Some(prev_neuron) = self.neurons.get(&prev_id) {
                if let Some(prev_weight) = prev_neuron.synaptic_weights.get(neuron_id) {
                    deflation += prev_neuron.membrane_potential * prev_weight;
                }
            }
        }

        for (i, weight) in neuron.synaptic_weights.iter_mut().enumerate() {
            let x = input_vector.get(i).unwrap_or(&0.0);
            let delta = self.config.a_plus * y * (*x - y * *weight - deflation);
            *weight = (*weight + delta * current_valence).clamp(self.config.min_weight, self.config.max_weight);
        }
    }

    pub fn get_novelty_drive(&self, neuron_id: &str) -> f64 {
        self.neurons.get(neuron_id)
            .map(|n| n.trace_pre.max(n.trace_post) * 0.4)
            .unwrap_or(0.0)
    }
}
