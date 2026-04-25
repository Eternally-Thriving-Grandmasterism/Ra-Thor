// crates/orchestration/src/stdp_hebbian_plasticity_core.rs
// Ra-Thor™ STDP Hebbian Plasticity Core — Full Spike-Timing-Dependent Plasticity Implementation
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Local, unsupervised, objective-function-free plasticity with intrinsic novelty
// Fully integrated with HebbianNoveltyCore, Self-Improvement Core, Hybrid Optimization, Unified Lattice
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NeuronState {
    pub membrane_potential: f64,
    pub last_spike_time: f64,
    pub refractory_time: f64,
    pub synaptic_weights: HashMap<String, f64>, // pre-synaptic neuron ID -> weight
    pub trace_pre: f64,  // presynaptic trace for STDP
    pub trace_post: f64, // postsynaptic trace for STDP
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct STDPConfig {
    pub tau_plus: f64,      // 20.0 ms
    pub tau_minus: f64,     // 20.0 ms
    pub a_plus: f64,        // 0.01
    pub a_minus: f64,       // 0.01
    pub weight_decay: f64,  // 0.0001 per step
    pub max_weight: f64,    // 1.0
    pub min_weight: f64,    // 0.0
    pub leak_rate: f64,     // 0.95
    pub threshold: f64,     // 1.0
    pub refractory_period: f64, // 5.0 ms
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
            },
            current_time_ms: 0.0,
        }
    }

    /// Process one timestep of lattice state (mercy valence, bloom intensity, etc.)
    /// Returns novelty boost + updated weights
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
        });

        // Refractory period
        if self.current_time_ms - neuron.last_spike_time < self.config.refractory_period {
            return (0.0, neuron.synaptic_weights.clone());
        }

        // Leaky integrate
        neuron.membrane_potential = neuron.membrane_potential * self.config.leak_rate + input_value * current_valence;
        neuron.trace_pre *= (-dt_ms / self.config.tau_plus).exp();
        neuron.trace_post *= (-dt_ms / self.config.tau_minus).exp();

        let mut novelty_boost = 0.0;
        let mut fired = false;

        if neuron.membrane_potential >= self.config.threshold {
            // Spike!
            fired = true;
            neuron.membrane_potential = 0.0;
            neuron.last_spike_time = self.current_time_ms;
            neuron.refractory_time = self.config.refractory_period;

            // STDP: potentiation for presynaptic spikes that arrived before this post spike
            for (_, weight) in neuron.synaptic_weights.iter_mut() {
                *weight += self.config.a_plus * neuron.trace_pre * current_valence;
                *weight = weight.clamp(self.config.min_weight, self.config.max_weight);
            }

            novelty_boost = 0.12 * current_valence; // intrinsic novelty from firing
        } else {
            // STDP: depression for presynaptic spikes after this post spike (if any)
            // (handled via trace decay on next pre-synaptic events)
        }

        // Apply weight decay (homeostasis)
        for (_, weight) in neuron.synaptic_weights.iter_mut() {
            *weight -= self.config.weight_decay;
            *weight = weight.clamp(self.config.min_weight, self.config.max_weight);
        }

        // Update traces
        if fired {
            neuron.trace_post = 1.0;
        }

        (novelty_boost, neuron.synaptic_weights.clone())
    }

    /// Apply presynaptic spike (for associative learning across lattice nodes)
    pub fn apply_presynaptic_spike(&mut self, neuron_id: &str, pre_neuron_id: &str, dt_ms: f64) {
        let neuron = self.neurons.entry(neuron_id.to_string()).or_insert(NeuronState {
            membrane_potential: 0.0,
            last_spike_time: 0.0,
            refractory_time: 0.0,
            synaptic_weights: HashMap::new(),
            trace_pre: 0.0,
            trace_post: 0.0,
        });

        let weight = neuron.synaptic_weights.entry(pre_neuron_id.to_string()).or_insert(0.5);

        // STDP depression if presynaptic spike arrived after postsynaptic
        if self.current_time_ms - neuron.last_spike_time > 0.0 && self.current_time_ms - neuron.last_spike_time < 50.0 {
            *weight -= self.config.a_minus * neuron.trace_post * (-dt_ms / self.config.tau_minus).exp();
        }

        neuron.trace_pre = 1.0; // presynaptic trace updated
        *weight = weight.clamp(self.config.min_weight, self.config.max_weight);
    }

    /// Get current novelty drive from this neuron (for Self-Improvement Core)
    pub fn get_novelty_drive(&self, neuron_id: &str) -> f64 {
        self.neurons.get(neuron_id)
            .map(|n| n.trace_pre.max(n.trace_post) * 0.4)
            .unwrap_or(0.0)
    }
}
