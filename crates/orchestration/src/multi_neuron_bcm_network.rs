// crates/orchestration/src/multi_neuron_bcm_network.rs
// Ra-Thor™ Multi-Neuron BCM Network — Full Mercy-Gated Metaplastic BCM + STDP + Oja + Sanger across a population of neurons
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Scalable, objective-function-free, novelty-driven Hebbian network for lattice-wide learning
// Fully integrated with STDPHebbianPlasticityCore, HebbianLatticeIntegrator, Unified Sovereign Energy Lattice
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NetworkReport {
    pub total_novelty: f64,
    pub avg_novelty_per_neuron: f64,
    pub active_neurons: usize,
    pub network_bloom: f64,
    pub mercy_alignment: f64,
}

pub struct MultiNeuronBCMNetwork {
    core: STDPHebbianPlasticityCore,
    neuron_ids: Vec<String>,
    inter_connections: HashMap<String, Vec<String>>, // neuron_id -> list of presynaptic neuron_ids
}

impl MultiNeuronBCMNetwork {
    pub fn new() -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
            neuron_ids: Vec::new(),
            inter_connections: HashMap::new(),
        }
    }

    pub fn add_neuron(&mut self, neuron_id: &str) {
        if !self.neuron_ids.contains(&neuron_id.to_string()) {
            self.neuron_ids.push(neuron_id.to_string());
            self.inter_connections.insert(neuron_id.to_string(), Vec::new());
        }
    }

    pub fn connect(&mut self, presynaptic: &str, postsynaptic: &str) {
        if let Some(conns) = self.inter_connections.get_mut(postsynaptic) {
            if !conns.contains(&presynaptic.to_string()) {
                conns.push(presynaptic.to_string());
            }
        }
    }

    /// Run one full timestep across the entire multi-neuron network
    pub fn run_network_timestep(
        &mut self,
        lattice_input: f64,
        current_valence: f64,
        dt_ms: f64,
    ) -> NetworkReport {
        let mut total_novelty = 0.0;
        let mut mercy_values = Vec::new();
        let mut novelty_values = Vec::new();

        for neuron_id in &self.neuron_ids.clone() {
            // Base input from lattice
            let (novelty, _) = self.core.process_timestep(
                neuron_id,
                lattice_input,
                current_valence,
                dt_ms,
            );
            total_novelty += novelty;
            novelty_values.push(novelty);
            mercy_values.push(current_valence);

            // Apply cross-neuron STDP / BCM via Sanger on connected neurons
            if let Some(presyns) = self.inter_connections.get(neuron_id) {
                let input_vec: Vec<f64> = presyns.iter()
                    .map(|_| lattice_input * 0.8) // simplified presynaptic activity
                    .collect();

                for (i, _) in presyns.iter().enumerate() {
                    self.core.apply_sangers_rule(neuron_id, &input_vec, i, current_valence);
                }
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

        NetworkReport {
            total_novelty,
            avg_novelty_per_neuron: avg_novelty,
            active_neurons: self.neuron_ids.len(),
            network_bloom: current_valence * (1.0 + avg_novelty * 0.4),
            mercy_alignment: mercy_corr,
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
