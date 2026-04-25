// crates/orchestration/src/recurrent_bcm_network.rs
// Ra-Thor™ Recurrent BCM Network — Full Mercy-Gated Metaplastic BCM + STDP + Oja + Sanger with Recurrent (Feedback) Connections
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Objective-function-free, novelty-driven recurrent Hebbian network for true associative memory & intrinsic creativity
// Fully integrated with MultiNeuronBCMNetwork, STDPHebbianPlasticityCore, HebbianLatticeIntegrator
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RecurrentReport {
    pub total_novelty: f64,
    pub avg_novelty_per_neuron: f64,
    pub active_neurons: usize,
    pub network_bloom: f64,
    pub mercy_alignment: f64,
    pub recurrent_activity: f64,
}

pub struct RecurrentBCMNetwork {
    core: STDPHebbianPlasticityCore,
    neuron_ids: Vec<String>,
    recurrent_connections: HashMap<String, Vec<String>>, // neuron_id -> list of recurrent (feedback) neuron_ids
    delayed_traces: HashMap<String, f64>,                // delayed activity for stable recurrence
}

impl RecurrentBCMNetwork {
    pub fn new() -> Self {
        Self {
            core: STDPHebbianPlasticityCore::new(),
            neuron_ids: Vec::new(),
            recurrent_connections: HashMap::new(),
            delayed_traces: HashMap::new(),
        }
    }

    pub fn add_neuron(&mut self, neuron_id: &str) {
        if !self.neuron_ids.contains(&neuron_id.to_string()) {
            self.neuron_ids.push(neuron_id.to_string());
            self.recurrent_connections.insert(neuron_id.to_string(), Vec::new());
            self.delayed_traces.insert(neuron_id.to_string(), 0.0);
        }
    }

    pub fn add_recurrent_connection(&mut self, from: &str, to: &str) {
        if let Some(conns) = self.recurrent_connections.get_mut(to) {
            if !conns.contains(&from.to_string()) {
                conns.push(from.to_string());
            }
        }
    }

    /// Run one full timestep with recurrent (feedback) dynamics
    pub fn run_recurrent_timestep(
        &mut self,
        lattice_input: f64,
        current_valence: f64,
        dt_ms: f64,
    ) -> RecurrentReport {
        let mut total_novelty = 0.0;
        let mut mercy_values = Vec::new();
        let mut novelty_values = Vec::new();
        let mut recurrent_sum = 0.0;

        for neuron_id in &self.neuron_ids.clone() {
            // Base input + recurrent feedback (using delayed trace for stability)
            let recurrent_input = *self.delayed_traces.get(neuron_id).unwrap_or(&0.0) * 0.6;
            let combined_input = lattice_input + recurrent_input;

            let (novelty, _) = self.core.process_timestep(
                neuron_id,
                combined_input,
                current_valence,
                dt_ms,
            );
            total_novelty += novelty;
            novelty_values.push(novelty);
            mercy_values.push(current_valence);
            recurrent_sum += recurrent_input;

            // Update delayed trace for next timestep (stable recurrence)
            let new_trace = combined_input * 0.7 + *self.delayed_traces.get(neuron_id).unwrap_or(&0.0) * 0.3;
            self.delayed_traces.insert(neuron_id.clone(), new_trace);

            // Apply cross-recurrent Sanger GHA
            if let Some(recurrents) = self.recurrent_connections.get(neuron_id) {
                let input_vec: Vec<f64> = recurrents.iter()
                    .map(|_| lattice_input * 0.7)
                    .collect();

                for (i, _) in recurrents.iter().enumerate() {
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

        RecurrentReport {
            total_novelty,
            avg_novelty_per_neuron: avg_novelty,
            active_neurons: self.neuron_ids.len(),
            network_bloom: current_valence * (1.0 + avg_novelty * 0.5),
            mercy_alignment: mercy_corr,
            recurrent_activity: recurrent_sum,
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
