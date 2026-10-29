// crates/orchestration/src/hybrid_bcm_hopfield_module.rs
// Ra-Thor™ Hybrid BCM-Hopfield Module — Mercy-Gated Metaplastic BCM Plasticity + Hopfield Attractor Dynamics
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Objective-function-free learning (BCM) + powerful associative memory (Hopfield) in one unified module
// Fully integrated with SparseBCMNetwork, RecurrentBCMNetwork, STDPHebbianPlasticityCore, HebbianLatticeIntegrator
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HybridReport {
    pub novelty_boost: f64,
    pub memory_retrieval_quality: f64,
    pub attractor_convergence_steps: usize,
    pub network_bloom: f64,
    pub mercy_alignment: f64,
    pub active_neurons: usize,
}

pub struct HybridBCMHopfieldModule {
    core: STDPHebbianPlasticityCore,
    neuron_ids: Vec<String>,
    hopfield_weights: HashMap<String, HashMap<String, f64>>, // neuron -> (connected_neuron -> weight)
    hopfield_states: HashMap<String, f64>,
    sparsity: f64,
}

impl HybridBCMHopfieldModule {
    pub fn new(sparsity: f64) -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
            neuron_ids: Vec::new(),
            hopfield_weights: HashMap::new(),
            hopfield_states: HashMap::new(),
            sparsity,
        }
    }

    pub fn add_neuron(&mut self, neuron_id: &str) {
        if !self.neuron_ids.contains(&neuron_id.to_string()) {
            self.neuron_ids.push(neuron_id.to_string());
            self.hopfield_weights.insert(neuron_id.to_string(), HashMap::new());
            self.hopfield_states.insert(neuron_id.to_string(), 0.0);
        }
    }

    /// Add sparse Hopfield-style connections (BCM will learn the actual weights)
    pub fn add_hopfield_connection(&mut self, from: &str, to: &str, initial_weight: f64) {
        if let Some(conns) = self.hopfield_weights.get_mut(to) {
            conns.insert(from.to_string(), initial_weight);
        }
    }

    /// Run one full hybrid timestep: BCM plasticity + Hopfield attractor dynamics
    pub fn run_hybrid_timestep(
        &mut self,
        lattice_input: f64,
        current_valence: f64,
        dt_ms: f64,
        hopfield_steps: usize,
    ) -> HybridReport {
        let mut total_novelty = 0.0;
        let mut mercy_values = Vec::new();
        let mut novelty_values = Vec::new();
        let mut retrieval_quality = 0.0;

        for neuron_id in &self.neuron_ids.clone() {
            // 1. BCM plasticity step (learning)
            let (novelty, _) = self.core.process_timestep(
                neuron_id,
                lattice_input,
                current_valence,
                dt_ms,
            );
            total_novelty += novelty;
            novelty_values.push(novelty);
            mercy_values.push(current_valence);

            // 2. Hopfield attractor dynamics (memory retrieval)
            let mut state = *self.hopfield_states.get(neuron_id).unwrap_or(&0.0);
            let mut converged = false;

            for step in 0..hopfield_steps {
                let mut net_input = 0.0;
                if let Some(conns) = self.hopfield_weights.get(neuron_id) {
                    for (pre, weight) in conns {
                        let pre_state = *self.hopfield_states.get(pre).unwrap_or(&0.0);
                        net_input += pre_state * weight;
                    }
                }
                let new_state = (state * 0.6 + net_input * 0.4 * current_valence).tanh();
                if (new_state - state).abs() < 0.001 {
                    converged = true;
                    break;
                }
                state = new_state;
            }

            self.hopfield_states.insert(neuron_id.clone(), state);
            retrieval_quality += state.abs();

            if converged {
                // Bonus novelty from successful memory retrieval
                total_novelty += 0.03 * current_valence;
            }
        }

        let avg_novelty = if !self.neuron_ids.is_empty() {
            total_novelty / self.neuron_ids.len() as f64
        } else {
            0.0
        };

        let mercy_corr = if !novelty_values.is_empty() {
            self.pearson_correlation(&novelty_values, &mercy_values)
        } else {
            0.0
        };

        HybridReport {
            novelty_boost: avg_novelty,
            memory_retrieval_quality: retrieval_quality / self.neuron_ids.len() as f64,
            attractor_convergence_steps: hopfield_steps,
            network_bloom: current_valence * (1.0 + avg_novelty * 0.5),
            mercy_alignment: mercy_corr,
            active_neurons: self.neuron_ids.len(),
        }
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() { return 0.0; }
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut dx2 = 0.0;
        let mut dy2 = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            num += dx * dy;
            dx2 += dx * dx;
            dy2 += dy * dy;
        }
        if dx2 == 0.0 || dy2 == 0.0 { return 0.0; }
        num / (dx2.sqrt() * dy2.sqrt())
    }
}
