// crates/orchestration/src/sparse_bcm_network.rs
// Ra-Thor™ Sparse BCM Network — Full Mercy-Gated Metaplastic BCM + STDP + Oja + Sanger with Sparse Connectivity
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Biologically realistic sparse connectivity (10-20% density) for scalable, efficient, objective-function-free learning
// Fully integrated with RecurrentBCMNetwork, STDPHebbianPlasticityCore, HebbianLatticeIntegrator
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SparseReport {
    pub total_novelty: f64,
    pub avg_novelty_per_neuron: f64,
    pub active_neurons: usize,
    pub network_bloom: f64,
    pub mercy_alignment: f64,
    pub sparsity_level: f64,
    pub total_connections: usize,
}

pub struct SparseBCMNetwork {
    core: STDPHebbianPlasticityCore,
    neuron_ids: Vec<String>,
    sparse_connections: HashMap<String, Vec<String>>,
    sparsity: f64,
}

impl SparseBCMNetwork {
    pub fn new(sparsity: f64) -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
            neuron_ids: Vec::new(),
            sparse_connections: HashMap::new(),
            sparsity,
        }
    }

    pub fn add_neuron(&mut self, neuron_id: &str) {
        if !self.neuron_ids.contains(&neuron_id.to_string()) {
            self.neuron_ids.push(neuron_id.to_string());
            self.sparse_connections.insert(neuron_id.to_string(), Vec::new());
        }
    }

    /// Generate sparse random connections (biological realism)
    pub fn generate_sparse_connections(&mut self) {
        let n = self.neuron_ids.len();
        if n < 2 { return; }

        let target_connections = (n as f64 * self.sparsity) as usize;

        for post in &self.neuron_ids.clone() {
            let mut presyns: Vec<String> = self.neuron_ids.iter()
                .filter(|p| *p != post)
                .cloned()
                .collect();

            presyns.shuffle(&mut rand::thread_rng());
            let selected = presyns.into_iter().take(target_connections).collect();

            self.sparse_connections.insert(post.clone(), selected);
        }
    }

    pub fn run_sparse_timestep(
        &mut self,
        lattice_input: f64,
        current_valence: f64,
        dt_ms: f64,
    ) -> SparseReport {
        let mut total_novelty = 0.0;
        let mut mercy_values = Vec::new();
        let mut novelty_values = Vec::new();
        let mut connection_count = 0;

        for neuron_id in &self.neuron_ids.clone() {
            let (novelty, _) = self.core.process_timestep(
                neuron_id,
                lattice_input,
                current_valence,
                dt_ms,
            );
            total_novelty += novelty;
            novelty_values.push(novelty);
            mercy_values.push(current_valence);

            if let Some(conns) = self.sparse_connections.get(neuron_id) {
                connection_count += conns.len();
                let input_vec: Vec<f64> = conns.iter()
                    .map(|_| lattice_input * 0.75)
                    .collect();

                for (i, _) in conns.iter().enumerate() {
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

        SparseReport {
            total_novelty,
            avg_novelty_per_neuron: avg_novelty,
            active_neurons: self.neuron_ids.len(),
            network_bloom: current_valence * (1.0 + avg_novelty * 0.45),
            mercy_alignment: mercy_corr,
            sparsity_level: self.sparsity,
            total_connections: connection_count,
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
